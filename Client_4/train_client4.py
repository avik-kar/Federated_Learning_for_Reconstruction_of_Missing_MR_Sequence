import os
import time
import copy
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm as tq
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable, Function

from model_utils import *
from dataloader_client4 import *

def ch_shuffle(x):
    shuffIdx1         = torch.from_numpy(np.random.randint(0,2,x.size(0)))
    shuffIdx2         = 1-shuffIdx1
    d_in              = torch.Tensor(x.size()).cuda()
    d_in[:,shuffIdx1] = x[:,0]
    d_in[:,shuffIdx2] = x[:,1]
    shuffLabel        = torch.cat((shuffIdx1.unsqueeze(1),shuffIdx2.unsqueeze(1)),dim=1)
    return d_in, shuffLabel

def train(args):
    
    torch.autograd.set_detect_anomaly(True)
    
    # get dataloader for training
    trainloader = data.DataLoader(dataLoader(is_transform=True, img_size=args.img_size), batch_size=args.b_sz, shuffle=True, pin_memory=True)
    
    # initialize models
    if args.adv_train == 'y':
        D1Net = Discriminator()
        D2Net = Discriminator()
        D3Net = Discriminator()
    SNet = SUMNet()
    PNet = PreNet()
    
    list_models = os.listdir(args.modelpath)
    # load previous round model state dicts to discriminators
    if 'Discriminators_client4.pt' in list_models and args.adv_train == 'y':
        discriminators = torch.load(args.modelpath+'Discriminators_client4.pt')
        D1Net.load_state_dict(discriminators['D1'])
        D2Net.load_state_dict(discriminators['D2'])
        D3Net.load_state_dict(discriminators['D3'])
    # load averaged model state dicts to generators
    if 'rx_client4.pt' in list_models:
        received_models = torch.load(args.modelpath+'rx_client4.pt')
        SNet.load_state_dict(received_models['SNet'])
        PNet.load_state_dict(received_models['PNet'])
    
    # send models to cuda if available
    if args.use_gpu:
        if args.adv_train == 'y':
            D1Net = D1Net.cuda()
            D2Net = D2Net.cuda()
            D3Net = D3Net.cuda()
        SNet = SNet.cuda()
        PNet = PNet.cuda()
    
    # define optimizers
    if args.adv_train == 'y':
        optimizer_D1 = optim.Adam(D1Net.parameters(), lr=args.d_lr)
        optimizer_D2 = optim.Adam(D2Net.parameters(), lr=args.d_lr)
        optimizer_D3 = optim.Adam(D3Net.parameters(), lr=args.d_lr)
    optimizer_S = optim.Adam(SNet.parameters(), lr=args.lr)
    optimizer_P = optim.Adam(PNet.parameters(), lr=args.lr)
    
    # define loss functions
    criterion_Rec_flair = nn.MSELoss()
    criterion_Rec_t1 = nn.MSELoss()
    criterion_Rec_t1ce = nn.MSELoss()
    criterion_feat_flair = nn.MSELoss()
    criterion_feat_t1 = nn.MSELoss()
    criterion_feat_t1ce = nn.MSELoss()
    if args.adv_train == 'y':
        criterion_D1 = nn.BCEWithLogitsLoss()
        criterion_D2 = nn.BCEWithLogitsLoss()
        criterion_D3 = nn.BCEWithLogitsLoss()
    
    start = time.time()
    trainReconLoss = []
    trainDisLoss = []
    
    for epoch in range(args.epoch):
        epochStart = time.time()
        
        # variable initialization to store losses and count batches
        Recon_Loss = 0
        if args.adv_train == 'y':
            Dis1_Loss = 0
            Dis2_Loss = 0
            Dis3_Loss = 0
        train_Batches = 0
        
        # enable train model if in any case disabled
        if args.adv_train == 'y':
            D1Net.train(True)
            D2Net.train(True)
            D3Net.train(True)
        SNet.train(True)
        PNet.train(True)
        
        for data_ in tq(trainloader):
            all_seq, _flair, _t1, _t1ce = data_
            
            # send input tensors to cuda if available
            if args.use_gpu:
                all_seq, _flair, _t1, _t1ce = all_seq.cuda().float(), _flair.cuda().float(), _t1.cuda().float(), _t1ce.cuda().float()
            
            #forward pass through generator
            with torch.no_grad():
                ip_agnostic_feat = PNet(Variable(all_seq))
                rec_images = SNet(Variable(ip_agnostic_feat))
            
            # Discriminator training
            if args.adv_train == 'y':
                # setting optimizer gradients to zero
                optimizer_D1.zero_grad()
                optimizer_D2.zero_grad()
                optimizer_D3.zero_grad()
                
                # shuffling of channels
                Ch_shuffle_data_1 = torch.cat((rec_images[:,0,:,:].unsqueeze(1), all_seq[:,0,:,:].unsqueeze(1)),1)
                Ch_shuffle_data_2 = torch.cat((rec_images[:,1,:,:].unsqueeze(1), all_seq[:,1,:,:].unsqueeze(1)),1)
                Ch_shuffle_data_3 = torch.cat((rec_images[:,2,:,:].unsqueeze(1), all_seq[:,2,:,:].unsqueeze(1)),1)
                
                D1_in, D1_label = ch_shuffle(Ch_shuffle_data_1)
                D1_out = D1Net(Variable(D1_in)).view(D1_in.size(0),-1)
                D2_in, D2_label = ch_shuffle(Ch_shuffle_data_2)
                D2_out = D2Net(Variable(D2_in)).view(D2_in.size(0),-1)
                D3_in, D3_label = ch_shuffle(Ch_shuffle_data_3)
                D3_out = D3Net(Variable(D3_in)).view(D3_in.size(0),-1)
                
                # loss calculation and weight update of discriminators
                if args.use_gpu:
                    D1_loss = criterion_D1(D1_out,D1_label.float().cuda())
                    D2_loss = criterion_D2(D2_out,D2_label.float().cuda())
                    D3_loss = criterion_D3(D3_out,D3_label.float().cuda())
                else:
                    D1_loss = criterion_D1(D1_out,D1_label)
                    D2_loss = criterion_D2(D2_out,D2_label)
                    D3_loss = criterion_D3(D3_out,D3_label)
                
                D1_loss.backward()
                optimizer_D1.step()
                
                D2_loss.backward()
                optimizer_D2.step()
                
                D3_loss.backward()
                optimizer_D3.step()
            
            # PreNet training to enforce similarity
            if args.enforce_similarity == 'y':
                
                optimizer_P.zero_grad()
                ip_agnostic_feat_flair = PNet(Variable(_flair))
                feat_loss_flair = criterion_feat_flair(ip_agnostic_feat_flair, ip_agnostic_feat)
                feat_loss_flair.backward()
                optimizer_P.step()
                
                optimizer_P.zero_grad()
                ip_agnostic_feat_t1 = PNet(Variable(_t1))
                feat_loss_t1 = criterion_feat_t1(ip_agnostic_feat_t1, ip_agnostic_feat)
                feat_loss_t1.backward()
                optimizer_P.step()
                
                optimizer_P.zero_grad()
                ip_agnostic_feat_t1ce = PNet(Variable(_t1ce))
                feat_loss_t1ce = criterion_feat_t1ce(ip_agnostic_feat_t1ce, ip_agnostic_feat)
                feat_loss_t1ce.backward()
                optimizer_P.step()
                
            # Generator Network trainning
            
            # setting optimizer gradients to zero
            if args.adv_train == 'y':
                optimizer_D1.zero_grad()
                optimizer_D2.zero_grad()
                optimizer_D3.zero_grad()
            optimizer_S.zero_grad()
            optimizer_P.zero_grad()
            
            # forward pass through generator
            ran_ar = np.random.choice(4,all_seq.shape[0])
            input_ = copy.deepcopy(all_seq)
            for i in range(all_seq.shape[0]):
                input_[i,ran_ar[i],:,:] = torch.zeros((args.img_size, args.img_size))
            
            ip_agnostic_feat = PNet(Variable(input_))
            rec_images = SNet(Variable(ip_agnostic_feat))
            
            # forward pass through discriminators
            if args.adv_train == 'y':
                Ch_shuffle_data_1 = torch.cat((rec_images[:,0,:,:].unsqueeze(1), all_seq[:,0,:,:].unsqueeze(1)),1)
                Ch_shuffle_data_2 = torch.cat((rec_images[:,1,:,:].unsqueeze(1), all_seq[:,1,:,:].unsqueeze(1)),1)
                Ch_shuffle_data_3 = torch.cat((rec_images[:,2,:,:].unsqueeze(1), all_seq[:,2,:,:].unsqueeze(1)),1)
                
                D1_in, D1_label = ch_shuffle(Ch_shuffle_data_1)
                D1_out = D1Net(Variable(D1_in)).view(D1_in.size(0),-1)
                D2_in, D2_label = ch_shuffle(Ch_shuffle_data_2)
                D2_out = D2Net(Variable(D2_in)).view(D2_in.size(0),-1)
                D3_in, D3_label = ch_shuffle(Ch_shuffle_data_3)
                D3_out = D3Net(Variable(D3_in)).view(D3_in.size(0),-1)
                
                # calculate discriminator losses and store
                if args.use_gpu:
                    D1_loss = criterion_D1(D1_out,D1_label.float().cuda())
                    D2_loss = criterion_D2(D2_out,D2_label.float().cuda())
                    D3_loss = criterion_D3(D3_out,D3_label.float().cuda())
                else:
                    D1_loss = criterion_D1(D1_out,D1_label)
                    D2_loss = criterion_D2(D2_out,D2_label)
                    D3_loss = criterion_D3(D3_out,D3_label)
                
                Dis1_Loss += D1_loss.item()
                Dis2_Loss += D2_loss.item()
                Dis3_Loss += D3_loss.item()
            
            # calculate generator losses
            Rec_loss_flair = criterion_Rec_flair(rec_images[:,0,:,:],all_seq[:,0,:,:])
            Rec_loss_t1 = criterion_Rec_t1(rec_images[:,1,:,:],all_seq[:,1,:,:])
            Rec_loss_t1ce = criterion_Rec_t1ce(rec_images[:,2,:,:],all_seq[:,2,:,:])
            
            # calculate total loss and store
            alpha = 0.001
            if args.adv_train == 'y':
                Total_Loss = (Rec_loss_flair + Rec_loss_t1 + Rec_loss_t1ce) - alpha*(D1_loss + D2_loss + D3_loss)
            else:
                Total_Loss = (Rec_loss_flair + Rec_loss_t1 + Rec_loss_t1ce)
            
            # update generator networks
            Total_Loss.backward()
            optimizer_S.step()
            optimizer_P.step()
            Recon_Loss += (Rec_loss_flair + Rec_loss_t1 + Rec_loss_t1ce).item()
            
            # setting optimizer gradients to zero
            if args.adv_train == 'y':
                optimizer_D1.zero_grad()
                optimizer_D2.zero_grad()
                optimizer_D3.zero_grad()
            optimizer_S.zero_grad()
            optimizer_P.zero_grad()

            train_Batches+=1
        
        # store epoch losses in a list
        avgReconLoss  = float(Recon_Loss)/train_Batches
        trainReconLoss.append(avgReconLoss)
        
        if args.adv_train == 'y':
            avgDisLoss     = float(Dis1_Loss + Dis2_Loss + Dis3_Loss)/train_Batches
            trainDisLoss.append(avgDisLoss)
        
        # print epoch info
        if args.adv_train == 'y':
            print("Epoch {}/{} completed in {:.2f} second | Avg Reconstruction loss: {:.4f} | Avg Discriminator loss: {:.4f}".format(epoch+1, args.epoch, time.time() - epochStart, avgReconLoss, avgDisLoss))
        else:
            print("Epoch {}/{} completed in {:.2f} second | Avg Reconstruction loss: {:.4f}".format(epoch+1, args.epoch, time.time() - epochStart, avgReconLoss))
    
    # store model weights
    transportables = {'SNet': SNet.state_dict(), 'PNet': PNet.state_dict(), 'client_info': ['flair', 't1', 't1ce']}
    
    torch.save(transportables, args.modelpath+'tx_client4.pt')
    
    if args.adv_train == 'y':
        discriminators = {'D1': D1Net.state_dict(), 'D2': D2Net.state_dict(), 'D3': D3Net.state_dict()}
        torch.save(discriminators, args.modelpath+'Discriminators_client4.pt')
    
    # loss plots
    plt.figure()
    plt.plot(range(1,len(trainReconLoss)+1), trainReconLoss, '-r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.xlim(1,len(trainReconLoss)+1)
    plt.savefig(args.savepath+'recon_loss.png')
    torch.save(trainReconLoss, args.savepath+'recon_loss.pt')
    del trainReconLoss, transportables, SNet, PNet, input_, all_seq, _t1, _t1ce, _flair, ip_agnostic_feat, rec_images, ip_agnostic_feat_t1, ip_agnostic_feat_t1ce, ip_agnostic_feat_flair
    
    if args.adv_train == 'y':
        plt.figure()
        plt.plot(range(1,len(trainDisLoss)+1), trainDisLoss, '-r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss')
        plt.xlim(1,len(trainDisLoss)+1)
        plt.savefig(args.savepath+'dis_loss.png')
        torch.save(trainDisLoss, args.savepath+'dis_loss.pt')
        del trainDisLoss, discriminators, D1Net, D2Net, D3Net
    
    print("Training ended in {:.2f} second.".format(time.time() - start)) 
    if args.use_gpu:
        torch.cuda.empty_cache()

def args_parser():
    parser = argparse.ArgumentParser()
    
    # training arguments
    parser.add_argument('--epoch', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--b_sz', type=int, default=16, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate for generator networks")
    parser.add_argument('--d_lr', type=float, default=1e-3, help="learning rate for discriminator networks")
    parser.add_argument('--img_size', type=int, default=256, help="image size")
    parser.add_argument('--adv_train', type=str, default='y', help="if 'y' then train adversarially, otherwise don't")
    parser.add_argument('--enforce_similarity', type=str, default='y', help="if 'y' then train PNet to have modality agnostic features at output")
    
    # model arguments
    parser.add_argument('--ftr_ch', type=int, default=16, help="number of channels in intermediate feature map")
    
    # other arguments
    parser.add_argument('--run_no', default=999, type=int, help='run number')
    
    args = parser.parse_args()
    
    args.use_gpu = torch.cuda.is_available()
    # directory to save results
    args.savepath = './Results/Run_'+str(args.run_no)+'/'
    args.modelpath = './Models/Run_'+str(args.run_no)+'/'
    
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    if not os.path.isdir(args.modelpath):
        os.makedirs(args.modelpath)
    # saving args as dict
    torch.save(vars(args), args.savepath+'args.pt',)
    return args

if __name__ == "__main__":
    
    args = args_parser()
    
    train(args)
