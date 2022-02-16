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
import pickle

def train_net(net_g,
              net_s1,
              net_s2,
              net_s0,
              device,
              epochs=5,
              batch_size=4,
              lr=0.1,
              first_num_of_kernels=64,
              dir_checkpoint='so_checkpoint/',
              dir_result='so_result/',
              dir_graphs='graphs/',
              optimizer_method = 'Adam',
              source='HeLa',
              size=128,
              cell='bt474',
              scaling_type='unet',
              opt_g=None,
              opt_s1=None,
              opt_s2=None,
              opt_s0=None,
              dis_measure=False,
              co_s=1,
              tri_train=False,
              ):

    path_w = f"{dir_result}output.txt"
    path_lossList = f"{dir_result}loss_list.pkl"

    with open(path_w, mode='w') as f:
        
        f.write('first num of kernels:{} \n'.format(first_num_of_kernels))
        f.write('optimizer method:{} \n'.format(optimizer_method))

    if opt_g == None:
        #not contrain
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
                lr=lr,
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
                lr=lr,
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
                lr=lr,
                #0.002
                betas=(0.9, 0.999),
                eps=1e-08,
                #default
                weight_decay=0,
                #0.0005
                amsgrad=False
            )
            # 三叉trainの場合
            if tri_train:
                opt_s0 = optim.Adam(
                    net_s0.parameters(),
                    lr=lr,
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

    """
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
    """

    if cell == 'bt474':
        target = 'shsy5y'
        trDir = 'train_and_test'
    elif cell == 'shsy5y':
        target = 'bt474'
        trDir = 'train_and_test'
    
    sourceDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{cell}'
    targetDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{target}'

    
    #load train images
    trsourceFiles = glob.glob(f'{sourceDir}/{trDir}/*')
    #print(trsourceFiles)
    trains_s = create_trainlist( trsourceFiles, scaling_type )

    #train: (520, 704)->(560, 784)
    for k in trains_s:
        k[0] = mirror_padding(k[0], 560, 784)
        k[1] = mirror_padding(k[1], 560, 784)
        
    #load val images
    valsourceFiles = glob.glob(f'{sourceDir}/val/*')
    valtargetFiles = glob.glob(f'{targetDir}/val/*')
    vals_s = create_trainlist( valsourceFiles, scaling_type )
    vals_t = create_trainlist( valtargetFiles, scaling_type )
        
    val_s = []
    val_t = []
        
    for l in vals_s:
        l[0] = mirror_padding(l[0], 544, 704)
        l[1] = mirror_padding(l[1], 544, 704)
        sepaList = cutting_img( l, 272, 352 )
        val_s.extend(sepaList)

    for l in vals_t:
        l[0] = mirror_padding(l[0], 544, 704)
        l[1] = mirror_padding(l[1], 544, 704)
        sepaList = cutting_img( l, 272, 352 )
        val_t.extend(sepaList)
        
    len_val_s = len(val_s)
    len_val_t = len(val_t)
    len_train = len(trains_s)

    tr_s_loss_list = []
    tr_d_loss_list = []
    val_s_loss_list = []
    val_d_loss_list = []    
    valdice_list = []
    min_val_s_loss = 10000.0;
    min_val_d_loss = 10000.0;
    
    
    for epoch in range(epochs):
        count = 0
        #train_s = ids_s['train']
        #val_s = ids_s['val']
        if cell == 'bt474' or 'shsy5y':
            #epochごとに(560, 784)から(272, 352) random crop & train_sに格納
            train_s = []
            for train_img_list in trains_s:
                train_s.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        
        #---- Train section

        s_epoch_loss = 0
        d_epoch_loss = 0
        
        for i, bs in enumerate(batch(train_s, batch_size, cell)):
            img_s = np.array([i[0] for i in bs]).astype(np.float32)
            mask = np.array([i[1] for i in bs]).astype(np.float32)

            img_s = torch.from_numpy(img_s).cuda(device)
            
            mask = torch.from_numpy(mask).cuda(device)
            mask_flat = mask.view(-1)
            
            #process1 ( g, s1 and s2 update )

            opt_g.zero_grad()
            opt_s1.zero_grad()
            opt_s2.zero_grad()
            
            feat_s =  net_g(img_s)
            mask_pred_s1 = net_s1(*feat_s)
            mask_pred_s2 = net_s2(*feat_s)
            if tri_train:
                opt_s0.zero_grad()
                mask_pred_s0 = net_s0(*feat_s)
                mask_prob_s0 = torch.sigmoid(mask_pred_s0)
                mask_prob_flat_s0 = mask_prob_s0.view(-1)
                loss_s0 = criterion(mask_prob_flat_s0, mask_flat)
            
            mask_prob_s1 = torch.sigmoid(mask_pred_s1)
            mask_prob_s2 = torch.sigmoid(mask_pred_s2)
            
            mask_prob_flat_s1 = mask_prob_s1.view(-1)
            mask_prob_flat_s2 = mask_prob_s2.view(-1)
            
            loss_s1 = criterion(mask_prob_flat_s1, mask_flat)
            loss_s2 = criterion(mask_prob_flat_s2, mask_flat)

            if dis_measure:
                loss_dis = torch.mean(torch.abs(mask_prob_flat_s1 - mask_prob_flat_s2))
                loss_s = loss_s1 + loss_s2 + co_s * loss_dis
                d_epoch_loss += loss_dis.item 
            else:
                loss_s = loss_s1 + loss_s2 if tri_train == False else loss_s1 + loss_s2 + loss_s0 

            loss_s.backward()

            #record segmentation loss 
            s_epoch_loss += loss_s.item()
            
            opt_g.step()
            opt_s1.step()
            opt_s2.step()
            if tri_train: opt_s0.step()
            
        seg = s_epoch_loss / (len_train/batch_size)
        dis = d_epoch_loss / (len_train/batch_size)
        
        with open(path_w, mode='a') as f:
            f.write('{} Epoch finished ! Loss: seg:{}\n'.format(epoch + 1, seg))    
        
        tr_s_loss_list.append(seg)
        tr_d_loss_list.append(dis)
        
        
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
                
                img_s =  torch.from_numpy(img_s).unsqueeze(0).cuda(device)
                mask = torch.from_numpy(mask).unsqueeze(0).cuda(device)
                
                mask_flat = mask.view(-1)

                #segmentation loss
                
                feat_s = net_g(img_s)
                mask_pred_s1 = net_s1(*feat_s)
                mask_pred_s2 = net_s2(*feat_s)
                
                mask_prob_s1 = torch.sigmoid(mask_pred_s1)
                mask_prob_s2 = torch.sigmoid(mask_pred_s2)
                
                mask_prob_flat_s1 = mask_prob_s1.view(-1)
                mask_prob_flat_s2 = mask_prob_s1.view(-1)
                
                if tri_train:
                    mask_pred_s0 = net_s0(*feat_s)
                    mask_prob_s0 = torch.sigmoid(mask_pred_s0)
                    mask_prob_flat_s0 = mask_prob_s0.view(-1)
                    loss_s0 = criterion(mask_prob_flat_s0, mask_flat)
                    
                loss_s1 = criterion(mask_prob_flat_s1, mask_flat)
                loss_s2 = criterion(mask_prob_flat_s2, mask_flat)
                
                loss = loss_s1 + loss_s2 if tri_train == False else loss_s1 + loss_s2 + loss_s0 

                val_s_loss += loss.item()

                #dice は一旦s1で計算
                mask_bin = (mask_prob_s1[0] > 0.5).float()
                val_dice += dice_coeff(mask_bin, mask.float(), device).item()

            for k, bt in enumerate(val_t):
                img_t = np.array(bt[0]).astype(np.float32)
                img_t = img_t.reshape([1, img_t.shape[-2], img_t.shape[-1]])
                img_t =  torch.from_numpy(img_t).unsqueeze(0).cuda(device)
                
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
            if tri_train:
                best_s0 = net_s0.state_dict()
                op_s0 = opt_s0.state_dict()
                torch.save({
                    'best_g' : best_g,
                    'best_s1' : best_s1,
                    'best_s2' : best_s2,
                    'best_s0' : best_s0,
                    'opt_g' : op_g,
                    'opt_s1' : op_s1,
                    'opt_s2' : op_s2,
                    'opt_s0' : op_s0,
                }, '{}CP_minSeg_sourceOnly_triTrain_e_{}'.format(dir_checkpoint, bestepoch))
            else:
                torch.save({
                    'best_g' : best_g,
                    'best_s1' : best_s1,
                    'best_s2' : best_s2,
                    'opt_g' : op_g,
                    'opt_s1' : op_s1,
                    'opt_s2' : op_s2,
                }, '{}CP_minseg_source_only_e_{}'.format(dir_checkpoint, bestepoch))
            with open(path_w, mode='a') as f:
                f.write(' best s model is updated \n')

        my_dict = {'tr_s_loss_list': tr_s_loss_list, 'val_s_loss_list': val_s_loss_list, 'val_d_loss_list': val_d_loss_list}
        with open(path_w, mode='a') as f:
            f.write('Validation Dice Coeff: {}\n'.format(valdice_list[bestepoch - 1]))

        with open(path_lossList, "wb") as tf:
            pickle.dump( my_dict, tf )

            
    # plot learning curve
    # segmentation loss
    draw_graph( dir_graphs, 'so_segmentation_loss', epochs, blue_list=tr_s_loss_list, blue_label='train', red_list=val_s_loss_list, red_label='validation' )
    # source discrepancy loss
    draw_graph( dir_graphs, 'source_trDisLoss', epochs, blue_list=tr_d_loss_list, blue_label='train' )
    # dice
    draw_graph( dir_graphs, 'dice', epochs, green_list=valdice_list,  green_label='validation_dice' )
    # discrepancy loss
    draw_graph( dir_graphs, 'so_discrepancy_loss', epochs, green_list=val_d_loss_list,  green_label='discrepancy' )
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
                        help='First num of kernels', dest='first_num_of_kernels')
    parser.add_argument('-om', '--optimizer-method', metavar='OM', type=str, nargs='?', default='Adam',
                        help='Optimizer method', dest='optimizer_method')
    parser.add_argument('-s', '--source', metavar='S', type=str, nargs='?', default='HeLa',
                        help='source cell', dest='source')
    parser.add_argument('-size', '--image-size', metavar='IS', type=int, nargs='?', default=128,
                        help='Image size', dest='size')
    parser.add_argument('-o', '--output', metavar='O', type=str, nargs='?', default='so_result',
                        help='out_dir?', dest='out_dir')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu_num?', dest='gpu_num')
    parser.add_argument('-cell', '--cell', metavar='CN', type=str, nargs='?', default='bt474',
                        help='cell name', dest='cell')
    parser.add_argument('-contrain', '--continue-training', metavar='CT', type=str, nargs='?', default=None,
                        help='load checkpoint path?', dest='contrain')
    parser.add_argument('-scaling', '--scaling-type', metavar='SM', type=str, nargs='?', default='unet',
                        help='scaling method??', dest='scaling_type')
    parser.add_argument('-dismeasure', '--discrepancy-measurement', metavar='DM', type=bool, nargs='?', default=False,
                        help='discrepancy measurement??', dest='dis_measure')
    parser.add_argument('-cos', '--coefficient-source-discrepancy', metavar='COS', type=float, default=1,
                        help='--coefficient-source-discrepancy', dest='co_s')
    parser.add_argument('-tri', '--tri-train', metavar='TT', type=bool, nargs='?', default=False,
                        help='tri train mode??', dest='tri_train')
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
    net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s2 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    
    net_g.to(device=device)
    net_s1.to(device=device)
    net_s2.to(device=device)

    if args.tri_train:
        net_s0 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_s0.to(device=device)
    else:
        net_s0 = None
    
    if args.contrain != None:
        checkPoint = torch.load(args.contrain)
        net_g.load_state_dict(checkPoint['best_g'])
        net_s1.load_state_dict(checkPoint['best_s1'])
        net_s2.load_state_dict(checkPoint['best_s2'])
        #optimizer = optim.Adam(model.parameters(), lr=1e-3)
        opt_s0 = None
        opt_g = optim.Adam(
            net_g.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )
        opt_s1 = optim.Adam(
            net_s1.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )
        opt_s2 = optim.Adam(
            net_s2.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )
        opt_g.load_state_dict(checkPoint['opt_g'])
        opt_s1.load_state_dict(checkPoint['opt_s1'])
        opt_s2.load_state_dict(checkPoint['opt_s2'])

        ###to cuda
        for state in opt_g.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device)
        for state in opt_s1.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device)
        for state in opt_s2.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device)
    else:
        
        opt_g = None
        opt_s1 = None
        opt_s2 = None
        opt_s0 = None
    
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    dir_result = './trResult/{}'.format(args.out_dir)
    dir_checkpoint = '{}/so_checkpoint'.format(dir_result)
    current_graphs = './graphs'
    dir_graphs = '{}/{}'.format(current_graphs, args.out_dir)
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(current_graphs, exist_ok=True)
    os.makedirs(dir_graphs, exist_ok=True)
    try:
        train_net(net_g=net_g,
                  net_s1=net_s1,
                  net_s2=net_s2,
                  net_s0=net_s0,
                  device=device,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  first_num_of_kernels=args.first_num_of_kernels,
                  dir_checkpoint=f'{dir_checkpoint}/',
                  dir_result=f'{dir_result}/',
                  dir_graphs=f'{dir_graphs}/',
                  optimizer_method=args.optimizer_method,
                  source=args.source,
                  size=args.size,
                  cell=args.cell,
                  scaling_type=args.scaling_type,
                  opt_g=opt_g,
                  opt_s1=opt_s1,
                  opt_s2=opt_s2,
                  opt_s0=opt_s0,
                  dis_measure=args.dis_measure,
                  co_s=args.co_s,
                  tri_train=args.tri_train,
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
