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
from model import UNet
import functions_io
from functions_io import * 

import glob
from skimage import io
import random
import matplotlib.pyplot as plt
#sys.path.append(os.getcwd())

def train_raw_net(net,
                  device,
                  epochs=1,
                  batch_size=16,
                  lr=0.0001,
                  first_num_of_kernels=64,
                  optimizer_method='Adam',
                  cell='bt474',
                  scaling_type='unet',
                  dir_checkpoint='checkpoint/',
                  dir_result='result/',
                  dir_graphs='graphs/'):

    # resultfile & losslist
    path_w = f"{dir_result}output.txt"
    path_lossList = f"{dir_result}loss_list.pkl"

    # recode training conditions
    with open( path_w, mode='w' ) as f:  
        f.write( 'first num of kernels:{} \n'.format( first_num_of_kernels ) )
        f.write( 'optimizer method:{}, learning rate:{} \n'.format( optimizer_method, lr ) )
        f.write( 'max epoch:{}, batchsize:{} \n'.format( epochs, batch_size ) )
        f.write( 'cell name:{} \n'.format( cell ) )

    # set the optimizer method
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
        optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )

    # loss function
    criterion = nn.BCELoss()

    # load train images
    cellDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{cell}'    
    trainFiles = glob.glob(f'{cellDir}/train/*')    
    trains = create_trainlist( trainFiles, scaling_type )
        
    # load val images
    valFiles = glob.glob(f'{cellDir}/val/*')
    vals = create_trainlist( valFiles, scaling_type )

    # train: (520, 704)->(560, 784)
    for k in trains:
        k[0] = mirror_padding(k[0], 560, 784)
        k[1] = mirror_padding(k[1], 560, 784)

    # val: (520, 704)->(544, 704)->(272, 352)*4
    val_sepa = []
    for l in vals:
        l[0] = mirror_padding(l[0], 544, 704)
        l[1] = mirror_padding(l[1], 544, 704)
        sepaList = cutting_img( l, 272, 352 )
        val_sepa.extend(sepaList)
    
    trloss_list = []
    valloss_list = []
    valiou_list = []
    min_val_loss = 10000.0
    len_train = len( trains )
    len_val = len( val_sepa )

    with open( path_w, mode='w' ) as f:  
        f.write(f"len_train is {len_train}\n")
        f.write(f"len_val is {len_val}\n")

    # fix the seed
    random.seed( 0 )
    
    for epoch in range(epochs):
        count = 0
        train = []

        for train_img_list in trains:
            train.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        
        random.shuffle(train)
        #---- Train section
        epoch_loss = 0
        for i, b in enumerate( batch( train, batch_size, cell ) ):
            
            img = np.array([i[0] for i in b]).astype(np.float32)
            mask = np.array([i[1] for i in b]).astype(np.float32)

            img = torch.from_numpy(img).cuda( device )
            mask = torch.from_numpy(mask).cuda( device )
            mask_flat = mask.view(-1)

            mask_pred = net(img)
            mask_prob = torch.sigmoid(mask_pred)
            
            mask_prob_flat = mask_prob.view(-1)
            
            #print('type of mask_prob_flat:', mask_prob_flat.dtype)
            #print('type of mask_flat:', mask_flat.dtype)
            loss = criterion(mask_prob_flat, mask_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
            #if i%10 == 0:
                #print('{}/{} ---- loss: {}'.format(i, int(len_train/batch_size), loss.item()))

        with open( path_w, mode='w' ) as f:  
            f.write('{} Epoch finished ! Loss: {}\n'.format(epoch + 1, epoch_loss / (len_train/batch_size)))

        print('{} Epoch finished ! Loss: {}'.format(epoch + 1, epoch_loss / (len_train/batch_size)))
        
        trloss_list.append( epoch_loss / (len_train / batch_size) )
        
        
        #---- Val section
        val_iou = 0
        val_loss = 0
        with torch.no_grad():
            for j, b in enumerate( val_sepa ):
                img = np.array(b[0]).astype(np.float32)
                img = img.reshape([1, img.shape[-2], img.shape[-1]])
                mask = np.array(b[1]).astype(np.float32)
                mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                img =  torch.from_numpy(img).unsqueeze(0).cuda( device )
                mask = torch.from_numpy(mask).unsqueeze(0).cuda( device )
                
                mask_flat = mask.view(-1)
                mask_pred = net(img)
                mask_prob = torch.sigmoid(mask_pred)
                mask_prob_flat = mask_prob.view(-1)
                loss = criterion(mask_prob_flat, mask_flat)
                val_loss += loss.item() 
                
                mask_bin = (mask_prob[0] > 0.5).float()
                val_iou = iou_loss(mask_bin, mask.float(), device).item()

        
        valloss_list.append(val_loss / len_val)
        valiou_list.append(val_iou / len_val)
        if (val_loss / len_val) < min_val_loss:
            min_val_loss = (val_loss / len_val)

            bestmodel = net.state_dict()
            bestepoch = epoch + 1
            torch.save(bestmodel, '{}CP_{}_{}_epoch{}_fk{}_b{}.pth'.format(dir_checkpoint, cell, optimizer_method, bestepoch, first_num_of_kernels, batch_size))

            with open( path_w, mode='w' ) as f:  
                f.write('best model is updated !\n')
                f.write('Checkpoint {}_epoch{}_fk{}_b{} saved !\n'.format(optimizer_method, bestepoch, first_num_of_kernels, batch_size))
                f.write('Validation IoU Loss: {}\n'.format(valiou_list[bestepoch - 1]))
    
    loss_graph = plt.figure()
    plt.plot(range(epochs), trloss_list, 'b-', label='train_loss')
    plt.plot(range(epochs), valloss_list, 'r-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    loss_graph.savefig('{}loss.pdf'.format(dir_result))

    iou_graph = plt.figure()
    plt.plot(range(epochs), valiou_list, 'g-', label='val_iou')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('iou')
    plt.grid()
    iou_graph.savefig('{}iou.pdf'.format(dir_result))
