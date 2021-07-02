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
              check_p,
              epochs=5,
              batch_size=4,
              lr=0.1,
              first_num_of_kernels=64,
              dir_checkpoint='checkpoint/',
              dir_result='result/',
              optimizer_method = 'Adam',
              source='HeLa',
              target='3T3',
              size=128,
              num_k=2):
             
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
        opt_g.load_state_dict(check_p['opt_g'])

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
        opt_s1.load_state_dict(check_p['opt_s1'])

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
        opt_s2.load_state_dict(check_p['opt_s2'])

    criterion = nn.BCELoss()
    criterion_d = Diff2d()
    name = "phase"
    
    trains_s = get_img_list(name, cell=source)
    trains_t = get_img_list(name, cell=target)
        
    random.seed(0)
    random.shuffle(trains_s)
    random.shuffle(trains_t)

    n_s = 72
    n_t = 47

    #for using 同数data of source と target 
    d = (len(trains_s) - n_s) - (len(trains_t) - n_t)
    ids_s = {'train': trains_s[:-n_s], 'val': trains_s[-n_s:]}
    tmp_t_tr = trains_t[:-n_t]
    tmp_t_val = trains_t[-n_t:]
    tmp_t_tr.extend(tmp_t_tr[:d])
    ids_t = {'train': tmp_t_tr, 'val': tmp_t_val }

    len_train = len(ids_s['train'])
    #len_train_t = len(ids_t['train'])
    len_val_s = len(ids_s['val'])
    len_val_t = len(ids_t['val'])
    tr_s_loss_list = []
    tr_d_loss_list = []
    tr_d_loss_list_B = []
    tr_d_loss_list_C = []
    val_s_loss_list = []
    val_d_loss_list = []
    valdice_list = []
    disc_loss_at_B = []
    disc_loss_at_C = []
    min_val_s_loss = 10000.0;
    min_val_d_loss = 10000.0;
    deltaloss_C_to_B = []
    
    
    with torch.no_grad():
        val_d_loss = 0
        for j, bt in enumerate(ids_t['val']):
        #discrepancy loss
            img_t = np.array(bt[0]).astype(np.float32)
            img_t = img_t.reshape([1, img_t.shape[-2], img_t.shape[-1]])
            img_t =  torch.from_numpy(img_t).unsqueeze(0).cuda()
            
            feat_t = net_g(img_t)
            mask_pred_s1 = net_s1(*feat_t)
            mask_pred_s2 = net_s2(*feat_t)
        
            mask_prob_s1 = torch.sigmoid(mask_pred_s1)
            mask_prob_s2 = torch.sigmoid(mask_pred_s2)
        
            mask_prob_flat_s1 = mask_prob_s1.view(-1)
            mask_prob_flat_s2 = mask_prob_s2.view(-1)
        
            loss_dis = criterion_d(mask_prob_flat_s1, mask_prob_flat_s2)
        
            val_d_loss += loss_dis.item()
        print(val_d_loss / len_val_t)
    
    
    for epoch in range(epochs):
        count = 0
        train_s = ids_s['train']
        random.shuffle(train_s)
        train_t = ids_t['train']
        random.shuffle(train_t)
        val_s = ids_s['val']
        val_t = ids_t['val']

        
        #---- Train section

        s_epoch_loss = 0
        d_epoch_loss = 0
        d_loss_at_B = 0
        d_loss_at_C = 0
        for i, (bs, bt) in enumerate(zip(batch(train_s, batch_size), batch_t(train_t, batch_size))):
            img_s = np.array([i[0] for i in bs]).astype(np.float32)
            mask = np.array([i[1] for i in bs]).astype(np.float32)
            img_t = np.array([i[0] for i in bt]).astype(np.float32)

            img_s = torch.from_numpy(img_s).cuda()
            img_t = torch.from_numpy(img_t).cuda()
            mask = torch.from_numpy(mask).cuda()
            mask_flat = mask.view(-1)
            """
            #process1 ( g, s1 and s2 update )
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

            #record segmentation loss 
            s_epoch_loss += loss_s.item()
            
            opt_g.step()
            opt_s1.step()
            opt_s2.step()
            """
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
            #segmentationloss 保存
            s_epoch_loss += loss_s.item()
            
            loss_dis = criterion_d(mask_prob_flat_t1, mask_prob_flat_t2)

            loss = loss_s - 0.2 * loss_dis
            
            d_loss_at_B += loss_dis.item()
            #d_epoch_loss_after_A += 0.2 * loss_dis.item()
            
            ###Step B時点でのdisc_loss保存###
            disc_loss_at_B.append( loss_dis.item() )
            ######

            loss.backward()

            opt_s1.step()
            opt_s2.step()            
            #process3 ( g update )

            for k in range(num_k):
                opt_g.zero_grad()
                opt_s1.zero_grad()
                opt_s2.zero_grad()
                
                feat_t = net_g(img_t)
                mask_pred_t1 = net_s1(*feat_t)
                mask_pred_t2 = net_s2(*feat_t)

                mask_prob_t1 = torch.sigmoid(mask_pred_t1)
                mask_prob_t2 = torch.sigmoid(mask_pred_t2)
            
                mask_prob_flat_t1 = mask_prob_t1.view(-1)
                mask_prob_flat_t2 = mask_prob_t2.view(-1)

                #場合によってはloss_dis定数倍も視野
                loss_dis = criterion_d(mask_prob_flat_t1, mask_prob_flat_t2)

                if k == 0:

                    d_loss_at_C += loss_dis.item()
                    ###Step C時点のdisc_loss保存###
                    disc_loss_at_C.append(loss_dis.item())
                    ######
                
                loss_dis.backward()

                opt_g.step()
            
            #record discrepancy loss
            d_epoch_loss += abs(loss_dis.item())
                
            count += 1
            #if i%10 == 0:
                #print('{}/{} ---- loss: {}'.format(i, int(len_train/batch_size), loss.item()))

        
        seg = s_epoch_loss / (len_train/batch_size)
        dis = d_epoch_loss / (len_train/batch_size)    
        d_mean_at_B = d_loss_at_B / (len_train/batch_size)
        d_mean_at_C = d_loss_at_C / (len_train/batch_size)    
        print('{} Epoch finished ! Loss: seg:{}, dis:{}'.format(epoch + 1, seg, dis))
        
        tr_s_loss_list.append(seg)
        tr_d_loss_list.append(dis)
        deltaloss_C_to_B.append( d_mean_at_B - d_mean_at_C )
        #tr_d_loss_list_C.append(dis_after_B)
        
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

                
            for j, bt in enumerate(val_t):
                #discrepancy loss
                img_t = np.array(bt[0]).astype(np.float32)
                img_t = img_t.reshape([1, img_t.shape[-2], img_t.shape[-1]])
                img_t =  torch.from_numpy(img_t).unsqueeze(0).cuda()
                
                feat_t = net_g(img_t)
                mask_pred_s1 = net_s1(*feat_t)
                mask_pred_s2 = net_s2(*feat_t)
                
                mask_prob_s1 = torch.sigmoid(mask_pred_s1)
                mask_prob_s2 = torch.sigmoid(mask_pred_s2)
                
                mask_prob_flat_s1 = mask_prob_s1.view(-1)
                mask_prob_flat_s2 = mask_prob_s2.view(-1)

                loss_dis = criterion_d(mask_prob_flat_s1, mask_prob_flat_s2)

                val_d_loss += abs(loss_dis.item()) 

                
        current_val_s_loss = val_s_loss / len_val_s
        current_val_d_loss = val_d_loss / len_val_t
        
        print('val_seg:', current_val_s_loss)
        print('val_dis:', val_d_loss / len_val_t)
        val_s_loss_list.append(current_val_s_loss)
        val_d_loss_list.append(current_val_d_loss)
        valdice_list.append(val_dice / len_val_s)
        
        if current_val_s_loss < min_val_s_loss:
            min_val_s_loss = current_val_s_loss
            
            bestepoch = epoch + 1
            
            
            
            print('best model is updated !')

        if current_val_d_loss < min_val_d_loss:
            min_val_d_loss = current_val_d_loss
            
            bestepoch = epoch + 1
                        
            print('best model is updated !')
        
        
        best_g = net_g.state_dict()
        best_s = net_s1.state_dict()
        torch.save(best_g, '{}_G_e_{}.pth'.format(dir_checkpoint, epoch))
        torch.save(best_s, '{}_S_e_{}.pth'.format(dir_checkpoint, epoch))


    dt_now = datetime.datetime.now()
    y = dt_now.year
    mon = dt_now.month
    d = dt_now.day
    h = dt_now.hour
    m = dt_now.minute

    ###iterationごとのデータを整理###
    B_array = np.array( disc_loss_at_B )
    C_array = np.array( disc_loss_at_C )
    B_array_roll = np.roll( B_array, -1 )
    
    B_to_C_list = ( C_array - B_array ).tolist()
    C_to_B_list = ( B_array_roll - C_array ).tolist()
    #rollの帳尻合わせ
    del B_to_C_list[-1]
    del C_to_B_list[-1]
    iterations = len( B_to_C_list )

    #torch.save(best_g, '{}CP_G_epoch{}_{}{}{}_{}{}.pth'.format(dir_checkpoint, bestepoch, y, mon, d, h, m))
    #torch.save(best_s, '{}CP_S_epoch{}_{}{}{}_{}{}.pth'.format(dir_checkpoint, bestepoch, y, mon, d, h, m))
    #print('Checkpoint saved !')
    print('Validation Dice Coeff: {}'.format(valdice_list[bestepoch - 1]))
    
    # plot learning curve
    loss_s_graph = plt.figure()
    plt.plot(range(epochs), tr_s_loss_list, 'r-', label='train_loss')
    plt.plot(range(epochs), val_s_loss_list, 'b-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    loss_s_graph.savefig('{}seg_loss_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))

    loss_d_graph = plt.figure()
    plt.plot(range(epochs), tr_d_loss_list, 'r-', label='train_loss')
    plt.plot(range(epochs), val_d_loss_list, 'b-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    loss_d_graph.savefig('{}dis_loss_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))
    
    dice_graph = plt.figure()
    plt.plot(range(epochs), valdice_list, 'g-', label='val_dice')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.grid()
    dice_graph.savefig('{}dice_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))

    BC_d_graph = plt.figure()
    plt.plot(range(iterations), B_to_C_list, 'b-', label='Step B')
    plt.plot(range(iterations), C_to_B_list, 'g-', label='Step C')
    plt.legend()
    plt.xlabel('itaration')
    plt.ylabel('dloss')
    plt.grid()
    BC_d_graph.savefig('{}BCdis_iter_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))

    BC_d_graph_epoch = plt.figure()
    plt.plot(range(epochs), deltaloss_C_to_B, 'g-', label='Step C')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('dloss')
    plt.grid()
    BC_d_graph_epoch.savefig('{}BCdis_epoch_{}{}{}_{}{}.pdf'.format(dir_result, y, mon, d, h, m))

    

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
    parser.add_argument('-t', '--target', metavar='T', type=str, nargs='?', default='3T3',
                        help='target cell', dest='target')
    parser.add_argument('-size', '--image-size', metavar='IS', type=int, nargs='?', default=128,
                        help='Image size', dest='size')
    parser.add_argument('-nk', '--num_k', metavar='NK', type=int, nargs='?', default=2,
                        help='how many steps to repeat the generator update', dest='num_k')
    parser.add_argument('-m', '--model', metavar='M', type=str, nargs='?', default=2,
                        help='state dict path?', dest='path_of_model')
    
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
    check_p = torch.load(args.path_of_model)
    net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_g.load_state_dict(check_p['best_g'])
    net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s1.load_state_dict(check_p['best_s1'])
    net_s2 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s2.load_state_dict(check_p['best_s2'])
    
    net_g.to(device=device)
    net_s1.to(device=device)
    net_s2.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    dir_checkpoint = './reload_checkpoint'
    dir_result = './reload_result'
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(dir_result, exist_ok=True)
    try:
        train_net(net_g=net_g,
                  net_s1=net_s1,
                  net_s2=net_s2,
                  check_p=check_p,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  first_num_of_kernels=args.first_num_of_kernels,
                  device=device,
                  dir_checkpoint='{}/'.format(dir_checkpoint),
                  dir_result='{}/'.format(dir_result),
                  optimizer_method=args.optimizer_method,
                  source=args.source,
                  target=args.target,
                  size=args.size,
                  num_k=args.num_k)
                  #img_scale=args.scale,
                  #val_percent=args.val / 100)
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        #logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
