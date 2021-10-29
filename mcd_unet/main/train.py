import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import os
import model
from model import *
from functions_io import * 
import glob
from skimage import io
import random
import matplotlib.pyplot as plt
import datetime

def train_net(net_g,
              net_s1,
              net_s2,
              device,
              epochs=5,
              batch_size=4,
              lr=0.1,
              first_num_of_kernels=64,
              thresh=0.01,
              dir_checkpoint='checkpoint/',
              dir_result='result/',
              dir_graphs=f'result/',
              optimizer_method = 'Adam',
              source='HeLa',
              target='3T3',
              size=128,
              num_k=2,
              co_B=0.1,
              large_flag=False,
              try_flag=False,
              ssl_flag=False):

    path_w = f"{dir_result}output.txt"

    with open(path_w, mode='w') as f:
        
        f.write('first num of kernels:{} \n'.format(first_num_of_kernels))
        f.write('optimizer method:{} \n'.format(optimizer_method))

    
    
    if optimizer_method == 'SGD':
        opt_g = optim.SGD(
            net_g.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=2e-5
        )

        opt_s1 = optim.SGD(
            net_s1.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=2e-5
        )

        opt_s2 = optim.SGD(
            net_s2.parameters(),
            #lr=0.001,
            lr=0.0001,
            momentum=0.9,
            weight_decay=2e-5
        )
        
    else:
        opt_g = optim.Adam(
            net_g.parameters(),
            #lr=0.001,
            lr=0.0001,
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
            lr=0.0001,
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
            lr=0.0001,
            #0.002
            betas=(0.9, 0.999),
            eps=1e-08,
            #default
            weight_decay=0,
            #0.0005
            amsgrad=False
        )

    criterion = nn.BCELoss()
    #criterion_d = Diff2d()
    name = "phase"
    
    trains_s = get_img_list(name, source, large_flag)
    trains_t = get_img_list(name, target, large_flag)

    random.seed(0)
    random.shuffle(trains_s)
    random.shuffle(trains_t)

    if large_flag:
        n_s = 1
        n_t = 1
    else:
        n_s = 124
        n_t = 77
        

    #for using 同数data of source と target 
    d = (len(trains_s) - n_s) - (len(trains_t) - n_t)
    ids_s = {'train': trains_s[:-n_s], 'val': trains_s[-n_s:]}
    tmp_t_tr = trains_t[:-n_t]
    tmp_t_val = trains_t[-n_t:]
    tmp_t_tr.extend(tmp_t_tr[:d])
    ids_t = {'train': tmp_t_tr, 'val': tmp_t_val }

    len_train = len(ids_s['train'])
    #len_train_t = len(ids_t['train'])
    #len_val_s = len(ids_s['val'])
    #len_val_t = len(ids_t['val'])
    tr_s_loss_list = []
    tr_s_loss_list_C = []
    tr_s_loss_list_B = []
    tr_d_loss_list = []
    tr_d_loss_list_C = []
    tr_d_loss_list_B = []
    val_s_loss_list = []
    val_d_loss_list = []
    val_iou_s1_list = []
    val_iou_s2_list = []
    val_iou_t1_list = []
    val_iou_t2_list = []
    valdice_list = []
    min_val_s_loss = 10000.0;
    min_val_d_loss = 10000.0;
    AtoB = []
    BtoC = []
    CtoA = []

    if large_flag:
        
        tmp_train_s = ids_s['train']
        tmp_train_t = ids_t['train']

        val_s = ids_s['val']
        val_t = ids_t['val']

        print(f"len tmp_train_s:{len(tmp_train_s)}")
        print(type(tmp_train_s[0]))
        print(tmp_train_s[0][0].shape)
        #####train画像を1400x1680にmirror padding
        #source
        for k in tmp_train_s:
            k[0] = mirror_padding(k[0], 1400, 1680)
            k[1] = mirror_padding(k[1], 1400, 1680)
        #target
        for k in tmp_train_t:
            k[0] = mirror_padding(k[0], 1400, 1680)
            k[1] = mirror_padding(k[1], 1400, 1680)

        for l in val_s:
            l[0] = mirror_padding(l[0], 1024, 1536)
            l[1] = mirror_padding(l[1], 1024, 1536)

        for l in val_t:
            l[0] = mirror_padding(l[0], 1024, 1536)
            l[1] = mirror_padding(l[1], 1024, 1536)

        #6分割( valの枚数 = 1 を想定 )
        val_s = cutting_img( val_s[0], size )
        val_t = cutting_img( val_t[0], size )
        len_val_s = len( val_s )
        len_val_t = len( val_t )
        #print( "len of val_s is {}".format( len_val_s ) )
        #print( "len of val_t is {}".format( len_val_t ) )
        
    else:

        train_s = ids_s['train']
        train_t = ids_t['train']

        val_s = ids_s['val']
        val_t = ids_t['val']

    if try_flag:
        print("\ntry run end ...")
        return 0

    for epoch in range(epochs):
        count = 0
        #train_s = ids_s['train']
        #train_t = ids_t['train']
        if large_flag:
            train_s = []
            train_t = []

            for train_img_list in tmp_train_s:
                train_s.append( random_cropping( train_img_list[0], train_img_list[1], size ) )
            for train_img_list in tmp_train_t:
                train_t.append( random_cropping( train_img_list[0], train_img_list[1], size ) )
            
        random.shuffle(train_s)
        random.shuffle(train_t)
        #val_s = ids_s['val']
        #val_t = ids_t['val']

        
        #---- Train section

        s_epoch_loss = 0
        s_epoch_loss_after_A = 0
        s_epoch_loss_after_B = 0
        d_epoch_loss = 0
        d_epoch_loss_after_A = 0
        d_epoch_loss_after_B = 0
        for i, (bs, bt) in enumerate(zip(batch(train_s, batch_size), batch(train_t, batch_size))):
            img_s = np.array([i[0] for i in bs]).astype(np.float32)
            mask = np.array([i[1] for i in bs]).astype(np.float32)
            img_t = np.array([i[0] for i in bt]).astype(np.float32)
            
            img_s = torch.from_numpy(img_s).cuda(device)
            img_t = torch.from_numpy(img_t).cuda(device)
            mask = torch.from_numpy(mask).cuda(device)
            mask_flat = mask.view(-1)
            
            #process1 ( g, s1 and s2 update )
            #learning segmentation task 
            opt_g.zero_grad()
            opt_s1.zero_grad()
            opt_s2.zero_grad()
            loss_s = 0
            
            feat_s =  net_g(img_s)
            mask_pred_s1 = net_s1(*feat_s)
            mask_pred_s2 = net_s2(*feat_s)
            
            mask_prob_s1 = torch.sigmoid(mask_pred_s1)
            mask_prob_s2 = torch.sigmoid(mask_pred_s2)
            
            mask_prob_flat_s1 = mask_prob_s1.view(-1)
            mask_prob_flat_s2 = mask_prob_s2.view(-1)
            
            loss_s += criterion(mask_prob_flat_s1, mask_flat)
            loss_s += criterion(mask_prob_flat_s2, mask_flat)
                
                
            loss_s.backward()

            opt_g.step()
            opt_s1.step()
            opt_s2.step()

            #record segmentation loss 
            s_epoch_loss += loss_s.item()

            
            #process2 (s1 and s2 update )
            opt_g.zero_grad()
            opt_s1.zero_grad()
            opt_s2.zero_grad()
            loss_s = 0
            
            feat_s = net_g(img_s)
            mask_pred_s1 = net_s1(*feat_s)
            mask_pred_s2 = net_s2(*feat_s)
            
            mask_prob_s1 = torch.sigmoid(mask_pred_s1)
            mask_prob_s2 = torch.sigmoid(mask_pred_s2)
            
            mask_prob_flat_s1 = mask_prob_s1.view(-1)
            mask_prob_flat_s2 = mask_prob_s2.view(-1)

            feat_t = net_g(img_t)
            mask_pred_t1 = net_s1(*feat_t)
            mask_pred_t2 = net_s2(*feat_t)

            mask_prob_t1 = torch.sigmoid(mask_pred_t1)
            mask_prob_t2 = torch.sigmoid(mask_pred_t2)
            
            mask_prob_flat_t1 = mask_prob_t1.view(-1)
            mask_prob_flat_t2 = mask_prob_t2.view(-1)
            
            loss_s += criterion(mask_prob_flat_s1, mask_flat)
            loss_s += criterion(mask_prob_flat_s2, mask_flat)
            
            #loss_dis = criterion_d(mask_prob_flat_t1, mask_prob_flat_t2)
            loss_dis = torch.mean(torch.abs(mask_prob_flat_t1 - mask_prob_flat_t2))
            loss = loss_s - co_B * loss_dis
            

            #stepB時点で計算したsegmentation loss
            s_epoch_loss_after_A += loss_s.item()
            d_epoch_loss_after_A += loss_dis.item()
            A_dis = loss_dis.item()
            
            loss.backward()

            opt_s1.step()
            opt_s2.step()
            #opt_g.step()

            opt_g.zero_grad()
            opt_s1.zero_grad()
            opt_s2.zero_grad()

            #process3 ( g update )

            for k in range(num_k):
                
                feat_t = net_g(img_t)
                mask_pred_t1 = net_s1(*feat_t)
                mask_pred_t2 = net_s2(*feat_t)

                mask_prob_t1 = torch.sigmoid(mask_pred_t1)
                mask_prob_t2 = torch.sigmoid(mask_pred_t2)
            
                mask_prob_flat_t1 = mask_prob_t1.view(-1)
                mask_prob_flat_t2 = mask_prob_t2.view(-1)

                #場合によってはloss_dis定数倍も視野
                if ssl_flag:
                    pseudo_lab_t1, pseudo_lab_t1, pseudo_dis_loss  = create_pseudo_label(mask_prob_flat_t1, \
                                                                                         mask_prob_flat_t2, T_dis=thresh )
                    L_seg1 = criterion(mask_prob_flat_t1, pseudo_lab_t1)
                    L_seg2 = criterion(mask_prob_flat_t2, pseudo_lab_t2)

                    loss = L_seg1 + L_seg2 + pseudo_dis_loss
                else:
                    
                    
                
                    loss = torch.mean(torch.abs(mask_prob_flat_t1 - mask_prob_flat_t2))
                
                    if k == 0 :
                        #print('Step B :' , loss_dis.item() / 2)
                        B_dis = abs(loss_dis.item())
                    
                        C_dis = abs(loss_dis.item())
                        s_epoch_loss_after_B += loss_s.item()
                        d_epoch_loss_after_B += abs(loss_dis.item())
                    elif k == (num_k - 1):
                        #print('Step C :' , loss_dis.item() / 2)
                        C_dis = abs(loss_dis.item())
                
                #loss = loss_s + 2 * loss_dis
                #loss.backward()
                loss.backward()
                
                opt_g.step()
                #opt_s1.step()
                #opt_s2.step()

                opt_g.zero_grad()
                opt_s1.zero_grad()
                opt_s2.zero_grad()
            #record discrepancy loss
            d_epoch_loss += abs(loss_dis.item()) 
            

            
            count += 1
            if i%50 == 0:
                print('epoch : {}, iter : {}, A : {}, B : {}, C : {}'.format(epoch+1, i+1, A_dis,  B_dis, C_dis))
        print('\n')
        seg = s_epoch_loss / (len_train/batch_size)
        print('seg : ',seg)
        dis = d_epoch_loss / (len_train/batch_size)    
        
        seg_after_A = s_epoch_loss_after_A / (len_train/batch_size)
        dis_after_A = d_epoch_loss_after_A / (len_train/batch_size)
        seg_after_B = s_epoch_loss_after_B / (len_train/batch_size)
        dis_after_B = d_epoch_loss_after_B / (len_train/batch_size)
        
        with open(path_w, mode='a') as f:
            f.write('epoch {}: seg:{}, dis:{} \n'.format(epoch + 1, seg, dis))
            
        tr_s_loss_list.append(seg)
        tr_s_loss_list_B.append(seg_after_A)
        tr_s_loss_list_C.append(seg_after_B)
        
        tr_d_loss_list.append(dis)
        tr_d_loss_list_B.append(dis_after_A)
        tr_d_loss_list_C.append(dis_after_B)
        AtoB.append(dis_after_A - dis)
        BtoC.append(dis_after_B - dis_after_A) 
        CtoA.append(dis - dis_after_B)
        #---- Val section
        val_dice = 0
        val_iou_s1 = 0
        val_iou_s2 = 0
        val_iou_t1 = 0
        val_iou_t2 = 0
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

                loss_s1 = criterion(mask_prob_flat_s1, mask_flat)
                loss_s2 = criterion(mask_prob_flat_s2, mask_flat)
                loss = loss_s1 + loss_s2

                val_s_loss += loss.item()

                #dice は一旦s1で計算
                
                mask_bin_s1 = (mask_prob_s1[0] > 0.5).float()
                mask_bin_s2 = (mask_prob_s2[0] > 0.5).float()
                val_iou_s1 += iou_loss(mask_bin_s1, mask.float(), device).item()
                val_iou_s2 += iou_loss(mask_bin_s2, mask.float(), device).item()
                #val_dice += dice_coeff(mask_bin, mask.float(), device).item()

            for k, bt in enumerate(val_t):
                #discrepancy loss
                img_t = np.array(bt[0]).astype(np.float32)
                img_t = img_t.reshape([1, img_t.shape[-2], img_t.shape[-1]])
                img_t =  torch.from_numpy(img_t).unsqueeze(0).cuda(device)
                ###
                mask = np.array(bt[1]).astype(np.float32)
                mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                mask = torch.from_numpy(mask).unsqueeze(0).cuda(device)
                mask_flat = mask.view(-1)
                ####
                
                feat_t = net_g(img_t)
                mask_pred_t1 = net_s1(*feat_t)
                mask_pred_t2 = net_s2(*feat_t)
                
                mask_prob_t1 = torch.sigmoid(mask_pred_t1)
                mask_prob_t2 = torch.sigmoid(mask_pred_t2)
                
                mask_prob_flat_t1 = mask_prob_t1.view(-1)
                mask_prob_flat_t2 = mask_prob_t2.view(-1)

                
                mask_bin_1 = (mask_prob_t1[0] > 0.5).float()
                mask_bin_2 = (mask_prob_t2[0] > 0.5).float()
                val_iou_t1 += iou_loss(mask_bin_1, mask.float(), device).item()
                val_iou_t2 += iou_loss(mask_bin_2, mask.float(), device).item()
                
                loss_dis = torch.mean(torch.abs(mask_prob_flat_t1 - mask_prob_flat_t2))
                val_d_loss += loss_dis.item()
                
            
                
        current_val_s_loss = val_s_loss / len_val_s
        current_val_d_loss = val_d_loss / len_val_t
        with open(path_w, mode='a') as f:
            f.write('val: val_seg: {}, val_dis : {} \n'.format(current_val_s_loss, current_val_d_loss))
            
        val_s_loss_list.append(current_val_s_loss)
        val_d_loss_list.append(current_val_d_loss)
        
        #valdice_list.append(val_dice / len_val_s)
        val_iou_s1_list.append( val_iou_s1 / len_val_s )
        val_iou_s2_list.append( val_iou_s2 / len_val_s )
        val_iou_t1_list.append( val_iou_t1 / len_val_t )
        val_iou_t2_list.append( val_iou_t2 / len_val_t )

        s_best_g = net_g.state_dict()
        s_best_s = net_s1.state_dict()
        torch.save(s_best_g, '{}CP_G_epoch{}.pth'.format(dir_checkpoint, epoch+1))
        torch.save(s_best_s, '{}CP_S_epoch{}.pth'.format(dir_checkpoint, epoch+1))
        
        if current_val_s_loss < min_val_s_loss:
            min_val_s_loss = current_val_s_loss
            
            s_bestepoch = epoch + 1
            with open(path_w, mode='a') as f:
                f.write('val seg loss is update \n')

        if current_val_d_loss < min_val_d_loss:
            min_val_d_loss = current_val_d_loss
            
            d_bestepoch = epoch + 1
            with open(path_w, mode='a') as f:
                f.write('val dis loss is update \n')
                
    dt_now = datetime.datetime.now()
    y = dt_now.year
    mon = dt_now.month
    d = dt_now.day
    h = dt_now.hour
    m = dt_now.minute
    
    #segmentation loss graph
    draw_graph( dir_graphs, 'segmentation_loss', epochs, blue_list=tr_s_loss_list, blue_label='train', red_list=val_s_loss_list, red_label='validation' )

    #discrepancy loss graph
    draw_graph( dir_graphs, 'discrepancy_loss', epochs, blue_list=tr_d_loss_list, blue_label='train', red_list=val_d_loss_list, red_label='validation' )

    #dice graph
    #draw_graph( dir_graphs, 'dice', epochs, green_list=val_dice_list,  green_label='validation_dice' )

    #source iou graph
    draw_graph( dir_graphs, 'source_IoU', epochs, blue_list=val_iou_s1_list,  blue_label='s1_IoU', green_list=val_iou_s2_list,  green_label='s2_IoU', y_label='IoU' )

    #target iou graph
    draw_graph( dir_graphs, 'target_IoU', epochs, red_list=val_iou_t1_list,  red_label='t1_IoU', green_list=val_iou_t2_list,  green_label='t2_IoU', y_label='IoU' )
    
    #ABC_s_graph
    draw_graph( dir_graphs, 'ABC_seg', epochs, blue_list=tr_s_loss_list_B, blue_label='Step B', red_list=tr_s_loss_list, red_label='Step A', green_list=tr_s_loss_list_C,  green_label='Step C', y_label='Δlos' )

    #ABC_d_graph
    draw_graph( dir_graphs, 'ABC_dis', epochs, blue_list=tr_d_loss_list_B, blue_label='Step B', red_list=tr_d_loss_list, red_label='Step A', green_list=tr_d_loss_list_C,  green_label='Step C', y_label='Δloss' )

    #A_to_B_to_C_grapf
    draw_graph( dir_graphs, 'A_to_B_to_C', epochs, blue_list=BtoC, blue_label='Step B to Step C', red_list=AtoB, red_label='Step A to Step B', green_list=CtoA,  green_label='Step C to Step A', y_label='Δloss' )
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
                        help='First num of kernels', dest='first_num_of_kernels')
    parser.add_argument('-om', '--optimizer-method', metavar='OM', type=str, nargs='?', default='Adam',
                        help='Optimizer method', dest='optimizer_method')
    parser.add_argument('-s', '--source', metavar='S', type=str, nargs='?', default='HeLa',
                        help='source cell', dest='source')
    parser.add_argument('-t', '--target', metavar='T', type=str, nargs='?', default='3T3',
                        help='target cell', dest='target')
    parser.add_argument('-size', '--image-size', metavar='IS', type=int, nargs='?', default=128,
                        help='Image size', dest='size')
    parser.add_argument('-nk', '--num_k', metavar='NK', type=int, nargs='?', default=2,
                        help='how many steps to repeat the generator update', dest='num_k')
    parser.add_argument('-o', '--output', metavar='O', type=str, nargs='?', default='result',
                        help='out_dir?', dest='out_dir')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu_num?', dest='gpu_num')
    parser.add_argument('-cob', '--co_stepB', type=float, nargs='?', default=0.1,
                        help='the coefficient in B?', dest='co_B')
    parser.add_argument('-lf', '--large-flag', type=bool, nargs='?', default=False,
                        help='Is img size large?', dest='large_flag')
    parser.add_argument('-try', '--try-flag', type=bool, nargs='?', default=False,
                        help='run on try mode?', dest='try_flag')
    parser.add_argument('-ssl', '--self-supervised', type=bool, nargs='?', default=False,
                        help='ssl mode?', dest='ssl_flag')
    parser.add_argument('-th', '--threshold', type=float, nargs='?', default=0.01,
                        help='ssl threshold?', dest='thresh')
    
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
        
    dir_result = './{}'.format(args.out_dir)
    dir_checkpoint = '{}/checkpoint'.format(dir_result)
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
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  first_num_of_kernels=args.first_num_of_kernels,
                  device=device,
                  thresh=args.thresh,
                  dir_checkpoint=f'{dir_checkpoint}/',
                  dir_result=f'{dir_result}/',
                  dir_graphs=f'{dir_graphs}/',
                  optimizer_method=args.optimizer_method,
                  source=args.source,
                  target=args.target,
                  size=args.size,
                  num_k=args.num_k,
                  co_B=args.co_B,
                  large_flag=args.large_flag,
                  try_flag=args.try_flag,
                  ssl_flag=args.ssl_flag)
                  
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
