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
import pickle
from train_raw_unet import train_raw_net

def train_net(net_1,
              net_2,
              lgnet_1,
              lgnet_2,
              device,
              epochs=5,
              batch_size=4,
              lr=0.0001,
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
              co_C=1.0,
              large_flag=False,
              try_flag=False,
              ssl_flag=False,
              scaling_type='normal',
              saEpoch=None,
              opt_1=None,
              opt_2=None,
              skipA=False,
              Bssl=False,
              pseConf=0,
              pseudo_mode=0,
              forget_rate=0.2,
              exponent=1,
              num_gradual=10,
              co_teaching=1,
              with_source=0):

    # resultfile & losslist
    path_w = f"{dir_result}output.txt"
    path_lossList = f"{dir_graphs}loss_list.pkl"
    
    # recode training conditions
    with open( path_w, mode='w' ) as f:  
        f.write( 'first num of kernels:{} \n'.format( first_num_of_kernels ) )
        f.write( 'optimizer method:{}, learning rate:{} \n'.format( optimizer_method, lr ) )
        f.write( 'source:{}, target:{} \n'.format( source, target ) )
        f.write( 'max epoch:{}, batchsize:{} \n'.format( epochs, batch_size ) )
        f.write( 'ssl_flag:{}, skipA:{} \n'.format( ssl_flag, skipA ) )
        f.write( 'co_B:{}, co_C:{}, num_k:{},  thresh:{} \n'.format( co_B, co_C, num_k, thresh ) )

    # optimizer set
    
    criterion = nn.BCELoss()
    
    sourceDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{source}'
    targetDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{target}'
    #load train images
    trsourceFiles = glob.glob(f'{sourceDir}/train/*')
    trtargetFiles = glob.glob(f'{targetDir}/cat_train/*')

    trains_s = create_trainlist( trsourceFiles, scaling_type )
    trains_t = create_trainlist( trtargetFiles, scaling_type )

    #train: (520, 704)->(560, 784)
    for k in trains_s:
        k[0] = mirror_padding( k[0], 560, 784 )
        k[1] = mirror_padding( k[1], 560, 784 )
    for k in trains_t:
        k[0] = mirror_padding( k[0], 560, 784 )
        k[1] = mirror_padding( k[1], 560, 784 )
    # adjust len(train_s) == len(train_t)
    if len( trains_s ) > len( trains_t ):
        d = len( trains_s ) - len( trains_t )
        trains_t.extend( trains_t[:d] )
    else:
        d = len( trains_t ) - len( trains_s )
        trains_s.extend( trains_s[:d] )
            
    len_train = len( trains_t )
    # load val images
    #valsourceFiles = glob.glob( f'{sourceDir}/val/*' )
    valtargetFiles = glob.glob( f'{targetDir}/val/*' )

    #vals_s = create_trainlist( valsourceFiles, scaling_type )
    vals_t = create_trainlist( valtargetFiles, scaling_type )

    #val_s = []
    val_t = []
    #for l in vals_s:
    #    l[0] = mirror_padding( l[0], 544, 704 )
    #    l[1] = mirror_padding( l[1], 544, 704 )
    #    sepaList = cutting_img( l, 272, 352 )
    #    val_s.extend( sepaList )
    for l in vals_t:
            l[0] = mirror_padding( l[0], 544, 704 )
            l[1] = mirror_padding( l[1], 544, 704 )
            sepaList = cutting_img( l, 272, 352 )
            val_t.extend( sepaList )

    #len_val_s = len( val_s )
    len_val_t = len( val_t )
        
    # s:segmentation d:discrepancy
    tr_s_loss_list_1 = []
    tr_s_loss_list_2 = []
    source_seg_list_1 = []
    source_seg_list_2 = []
    tgt_seg_list_1 = []
    tgt_seg_list_2 = []
    tr_s_loss_list_C = []
    tr_s_loss_list_B = []
    tr_d_loss_list = []
    tr_d_loss_list_C = []
    tr_d_loss_list_B = []
    val_iou_t_main_list = []
    valdice_list = []
    min_val_s_loss_1 = 10000.0;
    min_val_s_loss_2 = 10000.0;
    min_val_d_loss = 10000.0;
    AtoB = []
    BtoC = []
    CtoA = []
    pseudo_loss_list = []
    L_seg = 0
    assigned_list = []

    if try_flag:
        print(f'len_train_s: {len_train}')
        print(f'len_train_t: {len(trains_t)}')
        print(f'len_val_s: {len_val_s}')
        print(f'len_val_t: {len_val_t}')
        print("\ntry run end ...")
        return 0

    
    for epoch in range(epochs):
        
        count = 0
        train_s = []
        train_t = []

        for train_img_list in trains_s:
            train_s.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
            
        for train_img_list in trains_t:
            train_t.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )

        #random.shuffle( train_s )
        random.shuffle( train_s )
        random.shuffle( train_t )
        
        #---- Train section
        pseudo_loss = 0
        s_epoch_loss = 0
        d_epoch_loss = 0
        assignedSum = 0
        p_epoch_loss=0
        epoch_loss_1=0
        epoch_loss_2=0
        source_seg_epoch_loss_1 = 0
        source_seg_epoch_loss_2 = 0
        tgt_seg_epoch_loss_1 = 0
        tgt_seg_epoch_loss_2 = 0
        
        # sourceも用いる場合
        if with_source:
            for i, (bs, bt) in enumerate(zip(batch(train_s, batch_size, source), batch(train_t, batch_size, source))):

                # source data
                img_s = np.array([i[0] for i in bs]).astype(np.float32)
                img_s = torch.from_numpy(img_s).cuda(device)
                # source label
                mask = np.array([i[1] for i in bs]).astype(np.float32)
                mask = torch.from_numpy(mask).cuda(device)
                mask_flat = mask.view(-1)

                # target data  
                img_t = np.array([i[0] for i in bt]).astype(np.float32)
                img_t = torch.from_numpy(img_t).cuda(device)
                

                # generate pseudo label from lgnet_1 & lgnet_2 
                with torch.no_grad():
                    
                    lg1_pred = lgnet_1(img_t)
                    lg2_pred = lgnet_2(img_t)
                    
                    lg1_prob = torch.sigmoid(lg1_pred)
                    lg2_prob = torch.sigmoid(lg2_pred)

                    pseudo_prob = ( lg1_prob + lg1_prob ) / 2
                    pseudo_bin = ( pseudo_prob > 0.5 ).float()
                    pseudo_bin_flat = pseudo_bin.view(-1).detach()
                    
                # source output
                s_mask_pred_1 = net_1(img_s)
                s_mask_pred_2 = net_2(img_s)

                s_mask_prob_1 = torch.sigmoid(s_mask_pred_1)
                s_mask_prob_2 = torch.sigmoid(s_mask_pred_2)

                s_mask_prob_1_flat = s_mask_prob_1.view(-1)
                s_mask_prob_2_flat = s_mask_prob_2.view(-1)

                s_loss_1 = criterion( s_mask_prob_1_flat, mask_flat )
                s_loss_2 = criterion( s_mask_prob_2_flat, mask_flat )

                # target output
                mask_pred_1 = net_1(img_t)
                mask_pred_2 = net_2(img_t)

                mask_prob_1 = torch.sigmoid(mask_pred_1)
                mask_prob_2 = torch.sigmoid(mask_pred_2)

                mask_prob_1_flat = mask_prob_1.view(-1)
                mask_prob_2_flat = mask_prob_2.view(-1)

                t_loss_1 = criterion( mask_prob_1_flat, pseudo_bin_flat )
                t_loss_2 = criterion( mask_prob_2_flat, pseudo_bin_flat )
                loss_1 = s_loss_1 + t_loss_1
                loss_2 = s_loss_2 + t_loss_2
                
                opt_1.zero_grad()
                loss_1.backward()
                opt_1.step()
                opt_2.zero_grad()
                loss_2.backward()
                opt_2.step()

                source_seg_epoch_loss_1 += s_loss_1.item()
                source_seg_epoch_loss_2 += s_loss_2.item()

                tgt_seg_epoch_loss_1 += t_loss_1.item()
                tgt_seg_epoch_loss_2 += t_loss_2.item()

                epoch_loss_1 += loss_1.item()
                epoch_loss_2 += loss_2.item()
                count += 1
                if i != 0:
                    print('epoch : {}, iter : {}, loss_1 : {}, loss_2:{}'.format( epoch+1, i, epoch_loss_1/i, epoch_loss_2/i  ) )
        
        ### targetのみの場合
        else:
            for i, bt in enumerate( batch(train_t, batch_size, source) ):
                
                img_t = np.array([i[0] for i in bt]).astype(np.float32)
                img_t = torch.from_numpy(img_t).cuda(device)
                

                # generate pseudo label from lgnet_1 & lgnet_2 
                with torch.no_grad():
                    
                    lg1_pred = lgnet_1(img_t)
                    lg2_pred = lgnet_2(img_t)
                    
                    lg1_prob = torch.sigmoid(lg1_pred)
                    lg2_prob = torch.sigmoid(lg2_pred)

                    pseudo_prob = ( lg1_prob + lg1_prob ) / 2
                    pseudo_bin = ( pseudo_prob > 0.5 ).float()
                    pseudo_bin_flat = pseudo_bin.view(-1).detach()
                    
                    
                mask_pred_1 = net_1(img_t)
                mask_pred_2 = net_2(img_t)

                mask_prob_1 = torch.sigmoid(mask_pred_1)
                mask_prob_2 = torch.sigmoid(mask_pred_2)

                
                mask_prob_1_flat = mask_prob_1.view(-1)
                mask_prob_2_flat = mask_prob_2.view(-1)
                loss_1 = criterion( mask_prob_1_flat, pseudo_bin_flat )
                loss_2 = criterion( mask_prob_2_flat, pseudo_bin_flat )
                
                opt_1.zero_grad()
                loss_1.backward()
                opt_1.step()
                opt_2.zero_grad()
                loss_2.backward()
                opt_2.step()


                epoch_loss_1 += loss_1.item()
                epoch_loss_2 += loss_2.item()
                count += 1
                if i != 0:
                    print('epoch : {}, iter : {}, loss_1 : {}, loss_2:{}'.format( epoch+1, i, epoch_loss_1/i, epoch_loss_2/i ) )

        ### 全itr(epoch)終了 

                
        # epoch loss (seg & dis)
        seg_1 = epoch_loss_1 / (len_train/batch_size)
        seg_2 = epoch_loss_2 / (len_train/batch_size)
        
        
        with open(path_w, mode='a') as f:
            f.write('epoch {}: seg_1:{}, seg_2:{} \n'.format(epoch + 1, seg_1, seg_2))
            
        tr_s_loss_list_1.append(seg_1)
        tr_s_loss_list_2.append(seg_2)

        source_seg_list_1.append(source_seg_epoch_loss_1/(len_train/batch_size))
        source_seg_list_2.append(source_seg_epoch_loss_2/(len_train/batch_size))
        
        tgt_seg_list_1.append(tgt_seg_epoch_loss_1/(len_train/batch_size))
        tgt_seg_list_2.append(tgt_seg_epoch_loss_2/(len_train/batch_size))

        already = False
        if seg_1 < min_val_s_loss_1:
            min_val_s_loss_1 = seg_1
            
            s_bestepoch = epoch + 1
     
            best_net1 = net_1.state_dict()
            best_net2 = net_2.state_dict()
            
            best_opt1 = opt_1.state_dict()
            best_opt2 = opt_2.state_dict()
            
            torch.save({
                'best_net1' : best_net1,
                'best_net2' : best_net2,
                'opt_net1' : best_opt1,
                'opt_net2' : best_opt2,
            }, '{}CP_min_segloss1_e{}'.format(dir_checkpoint, epoch+1))

            already = True
            with open(path_w, mode='a') as f:
                f.write('seg loss 1 is update \n')

        if seg_2 < min_val_s_loss_2:

            min_val_s_loss_2 = seg_2
            s_bestepoch = epoch + 1

            if not already:
                best_net1 = net_1.state_dict()
                best_net2 = net_2.state_dict()
            
                best_opt1 = opt_1.state_dict()
                best_opt2 = opt_2.state_dict()
            
                torch.save({
                    'best_net1' : best_net1,
                    'best_net2' : best_net2,
                    'opt_net1' : best_opt1,
                    'opt_net2' : best_opt2,
                }, '{}CP_min_segloss1_e{}'.format(dir_checkpoint, epoch+1))

                with open(path_w, mode='a') as f:
                    f.write('seg loss 2 is update \n')            
                
        my_dict = { 'tr_s_loss_list_1': tr_s_loss_list_1, 'tr_s_loss_list_2': tr_s_loss_list_2, 'source_seg_list_1': source_seg_list_1, 'source_seg_list_2': source_seg_list_2, 'tgt_seg_list_1':tgt_seg_list_1, 'tgt_seg_list_2':tgt_seg_list_2  }

        with open(path_lossList, "wb") as tf:
            pickle.dump( my_dict, tf )
                
    
    #segmentation loss graph
    
    draw_graph( dir_graphs, 'train_loss', epochs, red_list=tr_s_loss_list_1,  red_label='loss1', blue_list=tr_s_loss_list_2, blue_label='loss2' )

    draw_graph( dir_graphs, 'train_source_loss', epochs, red_list=source_seg_list_1,  red_label='source loss1', blue_list=source_seg_list_2, blue_label='source loss2' )

    draw_graph( dir_graphs, 'train_target_loss', epochs, red_list=tgt_seg_list_1,  red_label='target loss1', blue_list=tgt_seg_list_2, blue_label='target loss2' )


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
    parser.add_argument('-s', '--source', metavar='S', type=str, nargs='?', default='bt474',
                        help='source cell', dest='source')
    parser.add_argument('-t', '--target', metavar='T', type=str, nargs='?', default='shsy5y',
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
    parser.add_argument('-coc', '--co_stepC', type=float, nargs='?', default=1.0,
                        help='the coefficient in C?', dest='co_C')
    parser.add_argument('-lf', '--large-flag', type=bool, nargs='?', default=False,
                        help='Is img size large?', dest='large_flag')
    parser.add_argument('-try', '--try-flag', type=bool, nargs='?', default=False,
                        help='run on try mode?', dest='try_flag')
    parser.add_argument('-ssl', '--self-supervised', type=bool, nargs='?', default=False,
                        help='ssl mode?', dest='ssl_flag')
    parser.add_argument('-th', '--threshold', type=float, nargs='?', default=0.01,
                        help='ssl threshold?', dest='thresh')
    parser.add_argument('-scaling', '--scaling-type', metavar='SM', type=str, nargs='?', default='unet',
                        help='scaling method??', dest='scaling_type')
    parser.add_argument('-saveall', '--saveall-epoch', metavar='SA', type=int, nargs='?', default=None,
                        help='epoch before which you save all models', dest='saEpoch')
    parser.add_argument('-contrain', '--continue-training', metavar='CT', type=str, nargs='?', default=None,
                        help='load checkpoint path?', dest='contrain')
    parser.add_argument('-net1', type=str, nargs='?', default=None,
                        help='load checkpoint path of net1', dest='net1')
    parser.add_argument('-net2', type=str, nargs='?', default=None,
                        help='load checkpoint path of net2', dest='net2')
    parser.add_argument('-skipA', '--skip-stepA', metavar='SKA', type=bool, nargs='?', default=False,
                        help='skip StepA?', dest='skipA')
    parser.add_argument('-Bssl', '--ssl-stepB', metavar='BSSL', type=bool, nargs='?', default=False,
                        help='use pseudo label in stepB?', dest='Bssl')
    parser.add_argument('-conf', '--pse-conf', metavar='PSEC', type=float, nargs='?', default=0.0,
                        help='the confidence of pseudo label?', dest='pseConf')
    parser.add_argument('-raw', '--raw-unet', type=bool, nargs='?', default=0,
                        help='train raw unet?', dest='raw_mode')
    parser.add_argument('-cell', type=str, nargs='?', default='bt474',
                        help='what cell you  use raw unet for?', dest='cell_raw')
    parser.add_argument('-fromterm2', type=bool, nargs='?', default=False,
                        help='retrain main network?', dest='fromterm2')
    parser.add_argument('-pseudo', type=int, nargs='?', default=0,
                        help='0:normal, 1:uncertainty from discrepancy, 3:distance map', dest='pseudo_mode')
    parser.add_argument('-preenco', type=int, nargs='?', default=1,
                        help='use pretrain encoder?', dest='pre_enco')
    parser.add_argument('-fr', type=float, nargs='?', default=None,
                        help='forget_rate ?', dest='forget_rate')
    parser.add_argument('-exp', type=float, nargs='?', default=1,
                        help='exponent 0.5 1 2?', dest='exponent')
    parser.add_argument('-ng', type=int, nargs='?', default=10,
                        help='num_gradual 5 10 15 ?', dest='num_gradual')
    parser.add_argument('-nl', type=float, nargs='?', default=0.2,
                        help='corruption rate, should be less than 1', dest='noise_rate')
    parser.add_argument('-coteaching', type=bool, nargs='?', default=1,
                        help='co_teaching or normal ?', dest='co_teaching')
    parser.add_argument('-next', type=bool, nargs='?', default=0,
                        help='pseudolab refine & model retrain ?', dest='next')
    parser.add_argument('-wsrc', type=bool, nargs='?', default=0,
                        help='train with source ?', dest='with_source')
    parser.add_argument('-seed', type=int, nargs='?', default=0,
                        help='seed num?', dest='seed')                    
    
  
    return parser.parse_args()

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


if __name__ == '__main__':
    args = get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_num}'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # fix the seed
    torch_fix_seed(args.seed)
    
    # co-teaching two networks
    net_1 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_1.to(device=device)
    net_2 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_2.to(device=device)

    # label generating two networks
    lgnet_1 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    lgnet_1.to(device=device)
    lgnet_2 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    lgnet_2.to(device=device)

    opt_1 = optim.Adam(
        net_1.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )
    
    opt_2 = optim.Adam(
        net_2.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )
    
    if args.next:
        # label refine & model retrain
        checkPoint = torch.load( args.net1, map_location=device )
        net_1.load_state_dict( checkPoint['best_net1'] )
        lgnet_1.load_state_dict( checkPoint['best_net1'] )
        net_2.load_state_dict( checkPoint['best_net2'] )
        lgnet_2.load_state_dict( checkPoint['best_net2'] )

        opt_1.load_state_dict(checkPoint['opt_net1'])
        opt_2.load_state_dict(checkPoint['opt_net2'])
        
    else:
        # loading models from checkpoints
        
        checkPoint_1 = torch.load( args.net1, map_location=device )
        net_1.load_state_dict( checkPoint_1['best_net'] )
        lgnet_1.load_state_dict( checkPoint_1['best_net'] )
        checkPoint_2 = torch.load( args.net2, map_location=device )
        net_2.load_state_dict( checkPoint_2['best_net'] )
        lgnet_2.load_state_dict( checkPoint_2['best_net'] )

        opt_1.load_state_dict(checkPoint_1['best_opt'])
        opt_2.load_state_dict(checkPoint_2['best_opt'])
    
            
    
    # optimizer to cuda
    for state in opt_1.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)

    for state in opt_2.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
        
    key = '' if args.raw_mode == False else '_raw'

    ### difine forget_rate
    if args.forget_rate is None:
        forget_rate=args.noise_rate
    else:
        forget_rate=args.forget_rate

    
    
    dir_result = './tr{}Result/{}'.format(key, args.out_dir)
    dir_checkpoint = '{}/checkpoint{}'.format(dir_result, key)
    current_graphs = f'./graphs{key}'
    dir_graphs = '{}/{}'.format(current_graphs, args.out_dir)
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(current_graphs, exist_ok=True)
    os.makedirs(dir_graphs, exist_ok=True)
    
    try:
        
        train_net(net_1=net_1,
                  net_2=net_2,
                  lgnet_1=lgnet_1,
                  lgnet_2=lgnet_2,
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
                  co_C=args.co_C,
                  large_flag=args.large_flag,
                  try_flag=args.try_flag,
                  ssl_flag=args.ssl_flag,
                  scaling_type=args.scaling_type,
                  saEpoch=args.saEpoch,
                  opt_1=opt_1,
                  opt_2=opt_2,
                  skipA=args.skipA,
                  Bssl=args.Bssl,
                  pseConf=args.pseConf,
                  pseudo_mode=args.pseudo_mode,
                  forget_rate=forget_rate,
                  exponent=args.exponent,
                  num_gradual=args.num_gradual,
                  co_teaching=args.co_teaching,
                  with_source=args.with_source)
        
            
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
