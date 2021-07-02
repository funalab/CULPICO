import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

import model
from model import *
from functions_io import * 
#from functions_io import scaling_image 
#from functions_io import dice_coeff
import glob
from skimage import io
import random
import matplotlib.pyplot as plt
#sys.path.append(os.getcwd())
import datetime

def train_net(net_g,
              net_s1,
              net_s2,
              device,
              epochs=5,
              batch_size=4,
              lr=0.1,
              first_num_of_kernels=64,
              dir_checkpoint='so_checkpoint/',
              dir_result='so_result/',
              optimizer_method = 'Adam',
              source='HeLa',
              size=128
              ):
             
    print('first num of kernels:', first_num_of_kernels)
    print('optimizer method:', optimizer_method)
    if optimizer_method == 'SGD':
        optimizer = optim.SGD(
            net.parameters(),
            lr=lr,
            #lr=0.1,
            momentum=0.9,
            #0.0005
            weight_decay=0.00005
        )
    else:
        opt_g = optim.Adam(
            net_g.parameters(),
            lr=0.001,
            #0.002
            betas=(0.9, 0.999),
            eps=1e-08,
            #default
            weight_decay=0,
            #0.0005
            amsgrad=False
        )
        opt_s1 = optim.Adam(
            net_s1.parameters(),
            lr=0.001,
            #0.002
            betas=(0.9, 0.999),
            eps=1e-08,
            #default
            weight_decay=0,
            #0.0005
            amsgrad=False
        )
        opt_s2 = optim.Adam(
            net_s2.parameters(),
            lr=0.001,
            #0.002
            betas=(0.9, 0.999),
            eps=1e-08,
            #default
            weight_decay=0,
            #0.0005
            amsgrad=False
        )

    criterion = nn.BCELoss()
    name = "phase"
    
    trains_s = get_img_list(name, cell=source)
    val_t = get_img_list(name, cell='3T3')    
    random.seed(0)
    random.shuffle(trains_s)

    n_s = 72
    
    ids_s = {'train': trains_s[:-n_s], 'val': trains_s[-n_s:]}
    
    len_train = len(ids_s['train'])
    
    len_val_s = len(ids_s['val'])
    
    tr_s_loss_list = []
    
    val_s_loss_list = []
    val_d_loss_list = []
    
    valdice_list = []
    min_val_s_loss = 10000.0;
    min_val_d_loss = 10000.0;
    
    
    for epoch in range(epochs):
        count = 0
        train_s = ids_s['train']
        val_s = ids_s['val']
        
        #---- Train section

        s_epoch_loss = 0
        
        for i, bs in enumerate(batch(train_s, batch_size)):
            img_s = np.array([i[0] for i in bs]).astype(np.float32)
            mask = np.array([i[1] for i in bs]).astype(np.float32)

            img_s = torch.from_numpy(img_s).cuda()
            
            mask = torch.from_numpy(mask).cuda()
            mask_flat = mask.view(-1)
            
            #process1 ( g, s1 and s2 update )

            opt_g.zero_grad()
            opt_s1.zero_grad()
            opt_s2.zero_grad()
            
            feat_s =  net_g(img_s)
            mask_pred_s1 = net_s1(*feat_s)
            mask_pred_s2 = net_s2(*feat_s)
            
            mask_prob_s1 = torch.sigmoid(mask_pred_s1)
            mask_prob_s2 = torch.sigmoid(mask_pred_s2)
            
            mask_prob_flat_s1 = mask_prob_s1.view(-1)
            mask_prob_flat_s2 = mask_prob_s2.view(-1)
            
            loss_s1 = criterion(mask_prob_flat_s1, mask_flat)
            loss_s2 = criterion(mask_prob_flat_s2, mask_flat)
            loss_s = loss_s1 + loss_s2

            loss_s.backward()

            #record segmentation loss 
            s_epoch_loss += loss_s.item()
            
            opt_g.step()
            opt_s1.step()
            opt_s2.step()
            
        seg = s_epoch_loss / (len_train/batch_size)
        
            
        print('{} Epoch finished ! Loss: seg:{}'.format(epoch + 1, seg))
        
        tr_s_loss_list.append(seg)
        
        
        
        #---- Val section
        val_dice = 0
        val_s_loss = 0
        val_d_loss = 0
        
        with torch.no_grad():
            for j, bs in enumerate(val_s):
                img_s = np.array(bs[0]).astype(np.float32)
                img_s = img_s.reshape([1, img_s.shape[-2], img_s.shape[-1]])
                mask = np.array(bs[1]).astype(np.float32)
                mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                
                img_s =  torch.from_numpy(img_s).unsqueeze(0).cuda()
                mask = torch.from_numpy(mask).unsqueeze(0).cuda()
                
                mask_flat = mask.view(-1)

                #segmentation loss
                
                feat_s = net_g(img_s)
                mask_pred_s1 = net_s1(*feat_s)
                mask_pred_s2 = net_s2(*feat_s)
                
                mask_prob_s1 = torch.sigmoid(mask_pred_s1)
                mask_prob_s2 = torch.sigmoid(mask_pred_s2)
                
                mask_prob_flat_s1 = mask_prob_s1.view(-1)
                mask_prob_flat_s2 = mask_prob_s1.view(-1)

                loss_s1 = criterion(mask_prob_flat_s1, mask_flat)
                loss_s2 = criterion(mask_prob_flat_s2, mask_flat)
                loss = loss_s1 + loss_s2

                val_s_loss += loss.item()

                #dice は一旦s1で計算
                mask_bin = (mask_prob_s1[0] > 0.5).float()
                val_dice += dice_coeff(mask_bin, mask.float()).item()

            for k, bt in enumerate(val_t):
                img_t = np.array(bt[0]).astype(np.float32)
                img_t = img_t.reshape([1, img_t.shape[-2], img_t.shape[-1]])
                img_t =  torch.from_numpy(img_t).unsqueeze(0).cuda()
                
                feat_t = net_g(img_t)
                mask_pred_t1 = net_s1(*feat_t)
                mask_pred_t2 = net_s2(*feat_t)
                
                mask_prob_t1 = torch.sigmoid(mask_pred_t1)
                mask_prob_t2 = torch.sigmoid(mask_pred_t2)
                
                mask_prob_flat_t1 = mask_prob_t1.view(-1)
                mask_prob_flat_t2 = mask_prob_t2.view(-1)

                #loss_dis = criterion_d(mask_prob_flat_t1, mask_prob_flat_t2)
                loss_dis = torch.mean(torch.abs(mask_prob_flat_t1 - mask_prob_flat_t2))
                val_d_loss += loss_dis.item()

        discrepancy = val_d_loss / len(val_t)
        val_d_loss_list.append(discrepancy)
             
        current_val_s_loss = val_s_loss / len_val_s
        val_s_loss_list.append(current_val_s_loss)
        valdice_list.append(val_dice / len_val_s)
        if current_val_s_loss < min_val_s_loss:
            min_val_s_loss = current_val_s_loss
            best_g = net_g.state_dict()
            best_s1 = net_s1.state_dict()
            best_s2 = net_s2.state_dict()
            op_g = opt_g.state_dict()
            op_s1 = opt_s1.state_dict()
            op_s2 = opt_s2.state_dict()
            bestepoch = epoch + 1
            print('best s model is updated ')
            
        if discrepancy < min_val_d_loss:
            min_val_d_loss = discrepancy
            d_best_g = net_g.state_dict()
            d_best_s1 = net_s1.state_dict()
            d_best_s2 = net_s2.state_dict()
            d_op_g = opt_g.state_dict()
            d_op_s1 = opt_s1.state_dict()
            d_op_s2 = opt_s2.state_dict()
            d_bestepoch = epoch + 1
            print('best d model is updated !')

    dt_now = datetime.datetime.now()
    y = dt_now.year
    mon = dt_now.month
    d = dt_now.day
    h = dt_now.hour
    m = dt_now.minute
    #torch.save(best_g, '{}CP_G_epoch{}_{}{}{}_{}{}.pth'.format(dir_checkpoint, bestepoch, y, mon, d, h, m))
    #torch.save(best_s, '{}CP_S_epoch{}_{}{}{}_{}{}.pth'.format(dir_checkpoint, bestepoch, y, mon, d, h, m))
    #print('Checkpoint saved !')
    """
    torch.save({
        'best_g' : best_g,
        'best_s1' : best_s1,
        'best_s2' : best_s2,
        'opt_g' : op_g,
        'opt_s1' : op_s1,
        'opt_s2' : op_s2,
    }, '{}CP_minseg_source_only_e_{}'.format(dir_checkpoint, bestepoch))

    torch.save({
        'best_g' : d_best_g,
        'best_s1' : d_best_s1,
        'best_s2' : d_best_s2,
        'opt_g' : d_op_g,
        'opt_s1' : d_op_s1,
        'opt_s2' : d_op_s2,
    }, '{}CP_mindis_source_only_e_{}'.format(dir_checkpoint, d_bestepoch))
    """
    print('Validation Dice Coeff: {}'.format(valdice_list[bestepoch - 1]))
    
    # plot learning curve
    loss_s_graph = plt.figure()
    plt.plot(range(epochs), tr_s_loss_list, 'r-', label='train_loss')
    plt.plot(range(epochs), val_s_loss_list, 'b-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    loss_s_graph.savefig('{}_so_seg_loss_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))

    dice_graph = plt.figure()
    plt.plot(range(epochs), valdice_list, 'g-', label='val_dice')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.grid()
    dice_graph.savefig('{}_dice_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))

    loss_d_graph = plt.figure()
    plt.plot(range(epochs), val_d_loss_list, 'g-', label='discrepancy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    loss_d_graph.savefig('{}_so_dis_loss_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=32,
                        help='First num of kernels', dest='first_num_of_kernels')
    parser.add_argument('-om', '--optimizer-method', metavar='OM', type=str, nargs='?', default='Adam',
                        help='Optimizer method', dest='optimizer_method')
    parser.add_argument('-s', '--source', metavar='S', type=str, nargs='?', default='HeLa',
                        help='source cell', dest='source')
    parser.add_argument('-size', '--image-size', metavar='IS', type=int, nargs='?', default=128,
                        help='Image size', dest='size')
    return parser.parse_args()

def get_img_list(name, cell):
    trains = []
    absolute = os.path.abspath('./dataset_smiyaki')    
    train_files = glob.glob(f"{absolute}/training_data/{cell}_set/*")
    for trainfile in train_files:
        ph_lab = [0] * 2
        #*set*/
        path_phase_and_lab = glob.glob(f"{trainfile}/*")
        #print(f"{trainfile}")
        for path_img in path_phase_and_lab:
            #print("hoge")
            img = io.imread(path_img)
            if name in path_img:
                #original unet scaling (subtracting by median)
                img = scaling_image(img)
                #img = img - np.median(img)

                #ndim==2でlistに格納
                ph_lab[0] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])

        trains.append(ph_lab)
    return trains

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s2 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    
    net_g.to(device=device)
    net_s1.to(device=device)
    net_s2.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    dir_checkpoint = './so_checkpoint'
    dir_result = './so_result'
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(dir_result, exist_ok=True)
    try:
        train_net(net_g=net_g,
                  net_s1=net_s1,
                  net_s2=net_s2,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  first_num_of_kernels=args.first_num_of_kernels,
                  device=device,
                  dir_checkpoint='so_checkpoint/',
                  dir_result='so_result/',
                  optimizer_method=args.optimizer_method,
                  source=args.source,
                  size=args.size,
                  )
                  #img_scale=args.scale,
                  #val_percent=args.val / 100)
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        #logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
