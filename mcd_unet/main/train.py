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
              skipA=False,
              Bssl=False,
              pseConf=0):

    # resultfile & losslist
    path_w = f"{dir_result}output.txt"
    path_lossList = f"{dir_graphs}loss_list.pkl"
    
    # recode training conditions
    with open( path_w, mode='w' ) as f:  
        f.write( 'first num of kernels:{} \n'.format( first_num_of_kernels ) )
        f.write( 'optimizer method:{}, learning rate:{} \n'.format( optimizer_method, lr ) )
        f.write( 'source:{}, target:{} \n'.format( source, target ) )
        f.write( 'max epoch:{}, batchsize:{} \n'.format( epochs, batch_size ) )
        f.write( 'co_B:{}, num_k:{} \n'.format( co_B, num_k ) )

    # optimizer set
    if opt_g == None:

        opt_g = optim.Adam(
            net_g.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
            )
        opt_s1 = optim.Adam(
            net_s1.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
            )
        opt_s2 = optim.Adam(
            net_s2.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
            )

    criterion = nn.BCELoss()
    

    sourceDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{source}'
    targetDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{target}'
    #load train images
    trsourceFiles = glob.glob(f'{sourceDir}/cat_train/*')
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
            
    len_train = len( trains_s )

    # load val images
    valtargetFiles = glob.glob( f'{targetDir}/val/*' )
    vals_t = create_trainlist( valtargetFiles, scaling_type )
    val_t = []
    for l in vals_t:
        l[0] = mirror_padding( l[0], 544, 704 )
        l[1] = mirror_padding( l[1], 544, 704 )
        sepaList = cutting_img( l, 272, 352 )
        val_t.extend( sepaList )    
    len_val_t = len( val_t )

    # s:segmentation d:discrepancy
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
    #random.seed( 0 )

    for epoch in range(epochs):

        #---- Create batch
        count = 0
        train_s = []
        train_t = []

        for train_img_list in trains_s:
            train_s.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        for train_img_list in trains_t:
            train_t.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )

        random.shuffle( train_s )
        random.shuffle( train_t )
                
        #---- Train section
        pseudo_loss = 0
        s_epoch_loss = 0
        d_epoch_loss = 0
        assignedSum = 0
        
        for i, (bs, bt) in enumerate(zip(batch(train_s, batch_size, source), batch(train_t, batch_size, source))):
            
            img_s = np.array([i[0] for i in bs]).astype(np.float32)
            mask = np.array([i[1] for i in bs]).astype(np.float32)
            img_t = np.array([i[0] for i in bt]).astype(np.float32)
            
            img_s = torch.from_numpy(img_s).cuda(device)
            img_t = torch.from_numpy(img_t).cuda(device)
            mask = torch.from_numpy(mask).cuda(device)
            mask_flat = mask.view(-1)
            
            ### process A ( g, s1 and s2 update ) ###
            # grad reset
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

            loss_s = 0
            loss_s += criterion(mask_prob_flat_s1, mask_flat)
            loss_s += criterion(mask_prob_flat_s2, mask_flat)

            # record segmentation loss 
            s_epoch_loss += loss_s.item()

            # parameter update
            loss_s.backward()
            opt_g.step()
            opt_s1.step()
            opt_s2.step()

            ### processB (s1 and s2 update ) ###
            # grad reset
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

            
            # loss_dis = criterion_d(mask_prob_flat_t1, mask_prob_flat_t2)
            
            # normal stepB loss (source segloss - target disloss )
            loss_dis = torch.mean(torch.abs(mask_prob_flat_t1 - mask_prob_flat_t2))
            loss = loss_s - co_B * loss_dis

            # parameter update
            loss.backward()
            opt_s1.step()
            opt_s2.step()

            ### processC ( g update ) ###
            
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

                loss_dis = torch.mean(torch.abs(mask_prob_flat_t1 - mask_prob_flat_t2))
                
                                    
                
                # not self supervised learning
                loss = co_C * loss_dis
                    
                # parameter update
                loss.backward()
                opt_g.step()

            #record discrepancy loss
            d_epoch_loss += loss_dis.item()


            count += 1
            if (i+1)%10 == 0:
                s_epoch_loss_iter = s_epoch_loss / i
                d_epoch_loss_iter = d_epoch_loss / i
                print('epoch : {}, iter : {}, seg_loss : {}, dis_loss :{}', epoch+1, i, s_epoch_loss_iter, d_epoch_loss_iter )

        # epoch loss (seg & dis)
        seg = s_epoch_loss / (len_train/batch_size)
        dis = d_epoch_loss / (len_train/batch_size)    
        
        
        with open(path_w, mode='a') as f:
            f.write('epoch {}: seg:{}, dis:{} \n'.format(epoch + 1, seg, dis))
            
        tr_s_loss_list.append(seg)
        tr_d_loss_list.append(dis)
        
        
        #---- Val section
        val_iou_t1 = 0
        val_iou_t2 = 0
        val_d_loss = 0

        with torch.no_grad():

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
                val_d_loss += co_C * loss_dis.item()
                 
                
        current_val_d_loss = val_d_loss / len_val_t
        with open(path_w, mode='a') as f:
            f.write('val_dis : {} \n'.format(current_val_d_loss))
            
        val_d_loss_list.append(current_val_d_loss)
        
        #valdice_list.append(val_dice / len_val_s)
        val_iou_t1_list.append( val_iou_t1 / len_val_t )
        val_iou_t2_list.append( val_iou_t2 / len_val_t )

       
        # minimum s_loss or d_loss 更新時checkpoint save
            
            
        if dis < min_val_d_loss:
            min_val_d_loss = dis

            
            best_g = net_g.state_dict()
            best_s1 = net_s1.state_dict()
            best_s2 = net_s2.state_dict()
            op_g = opt_g.state_dict()
            op_s1 = opt_s1.state_dict()
            op_s2 = opt_s2.state_dict()

            torch.save({
                'best_g' : best_g,
                'best_s1' : best_s1,
                'best_s2' : best_s2,
                'opt_g' : op_g,
                'opt_s1' : op_s1,
                'opt_s2' : op_s2,
            }, '{}CP_min_disloss_e{}'.format(dir_checkpoint, epoch+1))
            
            ###model, optimizer save
            
            with open(path_w, mode='a') as f:
                f.write('dis loss is update \n')

                
        my_dict = { 'tr_s_loss_list': tr_s_loss_list, 'tr_d_loss_list': tr_d_loss_list, 'val_d_loss_list': val_d_loss_list }

        with open(path_lossList, "wb") as tf:
            pickle.dump( my_dict, tf )
                
    
    #segmentation loss graph
    draw_graph( dir_graphs, 'segmentation_loss', epochs, red_list=tr_s_loss_list, red_label='train' )

    #discrepancy loss graph
    draw_graph( dir_graphs, 'discrepancy_loss', epochs, red_list=tr_d_loss_list, red_label='train' )

    #source iou graph
    draw_graph( dir_graphs, 'val_d_loss', epochs, blue_list=val_d_loss_list,  blue_label='validation' )

    #target iou graph
    draw_graph( dir_graphs, 'target_IoU', epochs, red_list=val_iou_t1_list,  red_label='t1_IoU', green_list=val_iou_t2_list,  green_label='t2_IoU', y_label='IoU' )
        
    

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
    parser.add_argument('-Lcell', type=str, nargs='?', default=None,
                        help='what cell you train sevenmodel for?', dest='Lcell_model')
    parser.add_argument('-seed', type=int, nargs='?', default=0,
                        help='seed num. ??', dest='seed')
    

    return parser.parse_args()

def torch_fix_seed(seed=42):
    # Python random & randam hash creation
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    #device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    torch_fix_seed(args.seed)
    
    if args.raw_mode == False:
        net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_s2 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_g.to(device=device)
        net_s1.to(device=device)
        net_s2.to(device=device)
        
        if args.contrain != None:
            checkPoint = torch.load(args.contrain)
            net_g.load_state_dict(checkPoint['best_g'])
            net_s1.load_state_dict(checkPoint['best_s1'])
            net_s2.load_state_dict(checkPoint['best_s2'])
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
    else:
        net = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net.to(device=device)
    
        
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
        if args.raw_mode == False:
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
                      co_C=args.co_C,
                      large_flag=args.large_flag,
                      try_flag=args.try_flag,
                      ssl_flag=args.ssl_flag,
                      scaling_type=args.scaling_type,
                      saEpoch=args.saEpoch,
                      opt_g=opt_g,
                      opt_s1=opt_s1,
                      opt_s2=opt_s2,
                      skipA=args.skipA,
                      Bssl=args.Bssl,
                      pseConf=args.pseConf)
        else:
            train_raw_net(
                net=net,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                first_num_of_kernels=args.first_num_of_kernels,
                device=device,
                optimizer_method=args.optimizer_method,
                cell=args.cell_raw,
                scaling_type=args.scaling_type,
                dir_checkpoint=f'{dir_checkpoint}/',
                dir_result=f'{dir_result}/',
                dir_graphs=f'{dir_graphs}/',
                Lcell_model=args.Lcell_model
            )
            
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
