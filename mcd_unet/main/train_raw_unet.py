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
#from functions_io import scaling_image 
#from functions_io import dice_coeff
import glob
from skimage import io
import random
import matplotlib.pyplot as plt
#sys.path.append(os.getcwd())

def train_net(net,
              device,
              epochs=5,
              batch_size=4,
              lr=0.1,
              first_num_of_kernels=64,
              dir_checkpoint='checkpoint/',
              dir_result='result/',
              optimizer_method = 'Adam',
              cell='HeLa',
              size=128):
              #val_percent=0.1,
              #save_cp=True,
              #img_scale=0.5

    #dataset = BasicDataset(dir_img, dir_mask, img_scale)
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = random_split(dataset, [n_train, n_val])
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    #global_step = 0
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
        optimizer = optim.Adam(
            net.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )
    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    criterion = nn.BCELoss()
    
    trains = []
    name = "phase"
    absolute = os.path.abspath('./dataset_smiyaki')
    if cell=='HeLa':
        #split train and validation
        if size == 128:
            n=72
        elif size == 640:
            n=4
        else:
            print('unexpected image size')
            
        print('training HeLa images')
    elif cell=='3T3':
        if size == 128:
            n=97
        elif size == 640:
            n=3
            
        else:
            print('unexpected image size')
            
        print('training 3T3 images')

    elif cell=='PNT':
        if size == 128:
            n=212
        else:
            print('unexpected image size')
    else:
        print('there is no data of cell named {}'.format(cell))

    if size == 128: 
        train_files = glob.glob(f"{absolute}/training_data/{cell}_set/*")
    elif size ==640:
        train_files = glob.glob(f"{absolute}/training_data/{cell}_640/*")
        
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
                ####z scoreに変更中！！
                img = standardize_image(img, True)
                #img = scaling_image(img)
                #img = img - np.median(img)

                #ndim==2でlistに格納
                ph_lab[0] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])

        trains.append(ph_lab)
        
    random.seed(0)
    random.shuffle(trains)
    
    #n = 72
    ids = {'train': trains[:-n], 'val': trains[-n:]}
    len_train = len(ids['train'])
    len_val = len(ids['val'])
    trloss_list = []
    valloss_list = []
    valdice_list = []
    min_val_loss = 10000.0;
    
    for epoch in range(epochs):
        count = 0
        train = ids['train']
        val = ids['val']
        random.shuffle(train)
        
        #---- Train section
        epoch_loss = 0
        for i, b in enumerate(batch(train, batch_size)):

            img = np.array([i[0] for i in b]).astype(np.float32)
            mask = np.array([i[1] for i in b]).astype(np.float32)

            img = torch.from_numpy(img).cuda()
            mask = torch.from_numpy(mask).cuda()
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
            
        print('{} Epoch finished ! Loss: {}'.format(epoch + 1, epoch_loss / (len_train/batch_size)))
        
        trloss_list.append(epoch_loss / (len_train / batch_size) )
        
        
        #---- Val section
        val_dice = 0
        val_loss = 0
        with torch.no_grad():
            for j, b in enumerate(val):
                img = np.array(b[0]).astype(np.float32)
                img = img.reshape([1, img.shape[-2], img.shape[-1]])
                mask = np.array(b[1]).astype(np.float32)
                mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                img =  torch.from_numpy(img).unsqueeze(0).cuda()
                mask = torch.from_numpy(mask).unsqueeze(0).cuda()
                ##############calculate validation loss#############################
                """
                valimg = np.array(b[0]).astype(np.float32)
                valmask = np.array(b[1]).astype(np.float32)

                valimg = torch.from_numpy(valimg).cuda()
                valmask = torch.from_numpy(valmask).cuda()
                valmask_flat = valmask.view(-1)

                valmask_pred = net(valimg)
                valmask_prob = F.sigmoid(valmask_pred)
            
                valmask_prob_flat = valmask_prob.view(-1)
            
                loss = criterion(valmask_prob_flat, valmask_flat)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                val_loss += loss.item() 
                ########################################################
                """
                mask_flat = mask.view(-1)
                mask_pred = net(img)
                mask_prob = torch.sigmoid(mask_pred)
                mask_prob_flat = mask_prob.view(-1)
                loss = criterion(mask_prob_flat, mask_flat)
                val_loss += loss.item() 
                
                mask_bin = (mask_prob[0] > 0.5).float()
                val_dice += dice_coeff(mask_bin, mask.float()).item()
                
                #if j == len_val - 1:
                #    print('{} validations completed'.format(len_val))
            
        valloss_list.append(val_loss / len_val)
        valdice_list.append(val_dice / len_val)
        if (val_loss / len_val) < min_val_loss:
            min_val_loss = (val_loss / len_val)
            bestmodel = net.state_dict()
            bestepoch = epoch + 1
            print('best model is updated !')
    torch.save(bestmodel, '{}CP_{}_{}_epoch{}_fk{}_b{}.pth'.format(dir_checkpoint, cell, optimizer_method, bestepoch, first_num_of_kernels, batch_size))
    print('Checkpoint {}_epoch{}_fk{}_b{} saved !'.format(optimizer_method, bestepoch, first_num_of_kernels, batch_size))
    print('Validation Dice Coeff: {}'.format(valdice_list[bestepoch - 1]))
    
    # plot learning curve
    loss_graph = plt.figure()
    plt.plot(range(epochs), trloss_list, 'r-', label='train_loss')
    plt.plot(range(epochs), valloss_list, 'b-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    loss_graph.savefig('{}loss.pdf'.format(dir_result))

    dice_graph = plt.figure()
    plt.plot(range(epochs), valdice_list, 'g-', label='val_dice')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.grid()
    dice_graph.savefig('{}dice.pdf'.format(dir_result))
    
####################################


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
    parser.add_argument('-cell', '--training-cell', metavar='TC', type=str, nargs='?', default='HeLa',
                        help='training cell image', dest='cell')
    parser.add_argument('-size', '--image-size', metavar='IS', type=int, nargs='?', default=128,
                        help='Image size', dest='size')
    
    #parser.add_argument('-f', '--load', dest='load', type=str, default=False,
    #                    help='Load model from a .pth file')
    #parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
    #                    help='Downscaling factor of the images')
    #parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
    #                    help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    dir_checkpoint = './checkpoint'
    dir_result = f'./result_{args.cell}_sz{args.size}_fk{args.first_num_of_kernels}_b{args.batchsize}_e{args.epochs}'
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(dir_result, exist_ok=True)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  first_num_of_kernels=args.first_num_of_kernels,
                  device=device,
                  dir_checkpoint='checkpoint/',
                  dir_result='result/',
                  optimizer_method=args.optimizer_method,
                  cell=args.cell,
                  size=args.size)
                  #img_scale=args.scale,
                  #val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        #logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
