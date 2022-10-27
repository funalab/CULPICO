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

def train_net(net_g,
              net_s1,
              net_s2,
              net_s_main,
              net_g_main,
              device,
              epochs=5,
              batch_size=4,
              lr=0.001,
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
              opt_g=None,
              opt_s1=None,
              opt_s2=None,
              opt_s_main=None,
              opt_g_main=None,
              skipA=False,
              Bssl=False,
              pseConf=0,
              pseudo_mode=0):

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
    if opt_g == None:
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
                lr=0.0001,
                momentum=0.9,
                weight_decay=2e-5
            )
        
        else:
            opt_g = optim.Adam(
                net_g.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                #0.0005
                amsgrad=False
            )
            opt_s1 = optim.Adam(
                net_s1.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                #0.0005
                amsgrad=False
            )
            opt_s2 = optim.Adam(
                net_s2.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                #0.0005
                amsgrad=False
            )

    criterion = nn.BCELoss()
        

    
    #sourceDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{source}'
    targetDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{target}'
    #load train images
    #trsourceFiles = glob.glob(f'{sourceDir}/train_and_test/*')
    trtargetFiles = glob.glob(f'{targetDir}/cat_train/*')

    #trains_s = create_trainlist( trsourceFiles, scaling_type )
    trains_t = create_trainlist( trtargetFiles, scaling_type )

    #train: (520, 704)->(560, 784)
    #for k in trains_s:
    #    k[0] = mirror_padding( k[0], 560, 784 )
    #    k[1] = mirror_padding( k[1], 560, 784 )
    for k in trains_t:
        k[0] = mirror_padding( k[0], 560, 784 )
        k[1] = mirror_padding( k[1], 560, 784 )
    # adjust len(train_s) == len(train_t)
    #if len( trains_s ) > len( trains_t ):
    #    d = len( trains_s ) - len( trains_t )
    #    trains_t.extend( trains_t[:d] )
    #else:
    #    d = len( trains_t ) - len( trains_s )
    #    trains_s.extend( trains_s[:d] )
            
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
    tr_s_loss_list = []
    tr_s_loss_list_C = []
    tr_s_loss_list_B = []
    tr_d_loss_list = []
    tr_d_loss_list_C = []
    tr_d_loss_list_B = []
    val_iou_t_main_list = []
    valdice_list = []
    min_val_s_loss = 10000.0;
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

    # fix the seed
    random.seed( 0 )
    for epoch in range(epochs):
        count = 0

        
        #train_s = []
        train_t = []

        #for train_img_list in trains_s:
        #    train_s.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        for train_img_list in trains_t:
            train_t.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )

        #random.shuffle( train_s )
        random.shuffle( train_t )
                
        #---- Train section
        pseudo_loss = 0
        s_epoch_loss = 0
        d_epoch_loss = 0
        assignedSum = 0
        p_epoch_loss=0
        
        for i, bt in enumerate( batch(train_t, batch_size, source) ):
            
            #img_s = np.array([i[0] for i in bs]).astype(np.float32)
            #mask = np.array([i[1] for i in bs]).astype(np.float32)
            img_t = np.array([i[0] for i in bt]).astype(np.float32)
            
            #img_s = torch.from_numpy(img_s).cuda(device)
            img_t = torch.from_numpy(img_t).cuda(device)
            #mask = torch.from_numpy(mask).cuda(device)
            #mask_flat = mask.view(-1)

            # generate pseudo label from s1 & s2 
            with torch.no_grad():
                feat_t = net_g(img_t)
                mask_pred_t1 = net_s1(*feat_t)
                mask_pred_t2 = net_s2(*feat_t)
                
                mask_prob_t1 = torch.sigmoid(mask_pred_t1)
                mask_prob_t2 = torch.sigmoid(mask_pred_t2)
                
                mask_prob_flat_t1 = mask_prob_t1.view(-1)
                mask_prob_flat_t2 = mask_prob_t2.view(-1)

                if pseudo_mode == 1:
                    pseudo_lab, confidence = create_uncer_pseudo( mask_prob_flat_t1, mask_prob_flat_t2, T_dis=thresh ,device=device )
                    pseudo_criterion = nn.BCELoss( weight=confidence )
                elif pseudo_mode == 2:
                    pseudo_lab, confidence = create_uncer_pseudo( mask_prob_flat_t1, mask_prob_flat_t2, T_dis=thresh ,device=device )
                    confidence = confidence * 0.8
                    pseudo_criterion = nn.BCELoss( weight=confidence )
                else:    
                    decide, pseudo_lab, assigned_C = create_pseudo_label(mask_prob_flat_t1, mask_prob_flat_t2,\
                                                                         T_dis=thresh, conf=pseConf, device=device)
                
                
            # calc loss for net_s_main
            feat_t = net_g_main(img_t)
            mask_pred_t_main = net_s_main(*feat_t)
            mask_prob_t_main = torch.sigmoid(mask_pred_t_main)
            mask_prob_flat_t_main = mask_prob_t_main.view(-1)

            if pseudo_mode == 1:
                loss = pseudo_criterion( mask_prob_flat_t_main, pseudo_lab.detach() )
            elif pseudo_mode == 2:
                loss = pseudo_criterion( mask_prob_flat_t_main, pseudo_lab.detach() )
            else:
                loss = criterion(mask_prob_flat_t_main[decide], pseudo_lab.detach())

            p_epoch_loss += loss.item()
            
            loss.backward()
            opt_g_main.step()
            opt_s_main.step()

            opt_g_main.zero_grad()
            opt_s_main.zero_grad()
            
            count += 1
            if (i+1)%10 == 0:
                p_epoch_loss_iter = p_epoch_loss / i
                print('epoch : {}, iter : {}, seg_loss : {}'.format( epoch+1, i, p_epoch_loss_iter ) )
        # epoch loss (seg & dis)
        seg = p_epoch_loss / (len_train/batch_size)
        
        
        with open(path_w, mode='a') as f:
            f.write('epoch {}: seg:{} \n'.format(epoch + 1, seg))
            
        tr_s_loss_list.append(seg)
        
        #---- Val section
        # none
        val_dice = 0
        val_iou_t_main = 0
        val_s_loss = 0
        val_d_loss = 0
        current_val_s_loss = 0
        current_val_d_loss = 0

        with torch.no_grad():
            for j, b in enumerate( val_t ):
                
                img = np.array(b[0]).astype(np.float32)
                img = img.reshape([1, img.shape[-2], img.shape[-1]])
                mask = np.array(b[1]).astype(np.float32)
                mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                img =  torch.from_numpy(img).unsqueeze(0).cuda( device )
                mask = torch.from_numpy(mask).unsqueeze(0).cuda( device )
                mask_flat = mask.view(-1)

                feat_t = net_g_main(img)

                mask_pred_t_main = net_s_main(*feat_t)
                
                mask_pred_prob_t_main = torch.sigmoid(mask_pred_t_main)
                mask_pred_prob_flat_t_main = mask_pred_prob_t_main.view(-1)

                mask_bin = (mask_pred_prob_t_main[0] > 0.5).float()
                
                val_iou_t_main += iou_loss(mask_bin, mask.float(), device).item()
                
        
        #val_s_loss_list.append(current_val_s_loss)
        #val_d_loss_list.append(current_val_d_loss)
        
        #valdice_list.append(val_dice / len_val_s)
        #val_iou_s1_list.append( val_iou_s1 / len_val_s )
        #val_iou_s2_list.append( val_iou_s2 / len_val_s )
        #val_iou_t1_list.append( val_iou_t1 / len_val_t )
        val_iou_t_main_list.append( val_iou_t_main / len_val_t )

        #s_best_g = net_g.state_dict()
        #s_best_s = net_s1.state_dict()
        #torch.save(s_best_g, '{}CP_G_epoch{}.pth'.format(dir_checkpoint, epoch+1))
        #torch.save(s_best_s, '{}CP_S_epoch{}.pth'.format(dir_checkpoint, epoch+1))
        
        # minimum s_loss or d_loss 更新時checkpoint saved 
        already = False
        if seg < min_val_s_loss:
            min_val_s_loss = seg
            #s_best_g = net_g.state_dict()
            #s_best_s = net_s1.state_dict()
            s_bestepoch = epoch + 1
            #torch.save(s_best_g, '{}CP_G_epoch{}.pth'.format(dir_checkpoint, epoch+1))
            #torch.save(s_best_s, '{}CP_S_epoch{}.pth'.format(dir_checkpoint, epoch+1))
     
            best_g_main = net_g_main.state_dict()
            #best_s1 = net_s1.state_dict()
            #best_s2 = net_s2.state_dict()
            best_s_main = net_s_main.state_dict()
            op_g_main = opt_g_main.state_dict()
            #op_s1 = opt_s1.state_dict()
            #op_s2 = opt_s2.state_dict()
            op_s_main = opt_s_main.state_dict()
            
            torch.save({
                'best_g_main' : best_g_main,
                'best_s_main' : best_s_main,
                #'best_s1' : best_s1,
                #'best_s2' : best_s2,
                'opt_g_main' : op_g_main,
                #'opt_s1' : op_s1,
                'opt_s_main' : op_s_main,
            }, '{}CP_min_segloss_e{}'.format(dir_checkpoint, epoch+1))

            already = True
            
            
            with open(path_w, mode='a') as f:
                f.write('val seg loss is update \n')
                        
            
                
        my_dict = { 'tr_s_loss_list': tr_s_loss_list, 'val_iou_t_main_list': val_iou_t_main_list  }

        with open(path_lossList, "wb") as tf:
            pickle.dump( my_dict, tf )
                
    
    #segmentation loss graph
    #draw_graph( dir_graphs, 'segmentation_loss', epochs, blue_list=tr_s_loss_list, blue_label='train', red_list=val_s_loss_list, red_label='validation' )
    draw_graph( dir_graphs, 'pseudo_loss', epochs, red_list=tr_s_loss_list,  red_label='train_loss' )

    draw_graph( dir_graphs, 'val_iou', epochs, green_list=val_iou_t_main_list,  green_label='val_iou' )

    """
    #discrepancy loss graph
    draw_graph( dir_graphs, 'discrepancy_loss', epochs, blue_list=tr_d_loss_list, blue_label='train', red_list=val_d_loss_list, red_label='validation' )

    #source iou graph
    draw_graph( dir_graphs, 'source_IoU', epochs, blue_list=val_iou_s1_list,  blue_label='s1_IoU', green_list=val_iou_s2_list,  green_label='s2_IoU', y_label='IoU' )

    #target iou graph
    draw_graph( dir_graphs, 'target_IoU', epochs, red_list=val_iou_t1_list,  red_label='t1_IoU', green_list=val_iou_t2_list,  green_label='t2_IoU', y_label='IoU' )
        
    # pseudo loss
    draw_graph( dir_graphs, 'pseudo_loss', epochs, red_list=pseudo_loss_list,  red_label='train_pseudo_loss' )

    # assigned percentage
    draw_graph( dir_graphs, 'assigned_percentage', epochs, green_list=assigned_list,  green_label='assigned_pseudo_label' )
    """

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
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
    

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_num}'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    
    net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s2 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s_main = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_g_main = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)

    net_g.to(device=device)
    net_s1.to(device=device)
    net_s2.to(device=device)
    net_s_main.to(device=device)
    net_g_main.to(device=device)
    
    checkPoint = torch.load(args.contrain)
    net_g.load_state_dict(checkPoint['best_g'])
    net_s1.load_state_dict(checkPoint['best_s1'])
    net_s2.load_state_dict(checkPoint['best_s2'])
    if args.pre_enco:
        net_g_main.load_state_dict(checkPoint['best_g'])
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
    opt_s_main = optim.Adam(
        net_s_main.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )
    opt_g_main = optim.Adam(
        net_g.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    if args.fromterm2:
        net_s_main.load_state_dict(checkPoint['best_s_main'])
        opt_s_main.load_state_dict(checkPoint['opt_s_main'])
        
    
    opt_g.load_state_dict(checkPoint['opt_g'])
    opt_s1.load_state_dict(checkPoint['opt_s1'])
    opt_s2.load_state_dict(checkPoint['opt_s2'])
    if args.pre_enco:
        opt_g_main.load_state_dict(checkPoint['opt_g'])

    
    
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
    if args.pre_enco:
        for state in opt_g_main.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device)
        
    key = '' if args.raw_mode == False else '_raw'
    
    dir_result = './tr{}Result/{}'.format(key, args.out_dir)
    dir_checkpoint = '{}/checkpoint{}'.format(dir_result, key)
    current_graphs = f'./graphs{key}'
    dir_graphs = '{}/{}'.format(current_graphs, args.out_dir)
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(current_graphs, exist_ok=True)
    os.makedirs(dir_graphs, exist_ok=True)
    
    try:
        
        train_net(net_g=net_g,
                  net_s1=net_s1,
                  net_s2=net_s2,
                  net_s_main=net_s_main,
                  net_g_main=net_g_main,
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
                  opt_g=opt_g,
                  opt_s1=opt_s1,
                  opt_s2=opt_s2,
                  opt_s_main=opt_s_main,
                  opt_g_main=opt_g_main,
                  skipA=args.skipA,
                  Bssl=args.Bssl,
                  pseConf=args.pseConf,
                  pseudo_mode=args.pseudo_mode)
        
            
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
