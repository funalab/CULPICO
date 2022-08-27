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

def train_net(net_main,
              net_aux,
              net_main_pseudo,
              net_aux_pseudo,
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
              opt_main=None,
              opt_aux=None,
              skipA=False,
              Bssl=False,
              pseConf=0):

    # resultfile & losslist
    path_w = f"{dir_result}output.txt"
    path_lossList = f"{dir_result}loss_list.pkl"
    
    # recode training conditions
    with open( path_w, mode='w' ) as f:  
        f.write( 'first num of kernels:{} \n'.format( first_num_of_kernels ) )
        f.write( 'optimizer method:{}, learning rate:{} \n'.format( optimizer_method, lr ) )
        f.write( 'source:{}, target:{} \n'.format( source, target ) )
        f.write( 'max epoch:{}, batchsize:{} \n'.format( epochs, batch_size ) )
        f.write( 'ssl_flag:{}, skipA:{} \n'.format( ssl_flag, skipA ) )
        f.write( 'co_B:{}, co_C:{}, num_k:{},  thresh:{} \n'.format( co_B, co_C, num_k, thresh ) )


    criterion = nn.BCELoss()
    #kldiv = nn.KLDivLoss(size_average=False)
    kldiv = nn.KLDivLoss(reduction="batchmean")
    #kldiv = nn.KLDivLoss(reduction="sum")
    
    if source == 'HeLa':
        name = "phase"
        trains_s = get_img_list(name, source, large_flag)
        trains_t = get_img_list(name, target, large_flag)

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
        

    elif source == 'bt474' or 'shsy5y':
        sourceDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{source}'
        targetDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{target}'
        #load train images
        trsourceFiles = glob.glob(f'{sourceDir}/train_and_test/*')
        trtargetFiles = glob.glob(f'{targetDir}/train/*')

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
        valsourceFiles = glob.glob( f'{sourceDir}/val/*' )
        valtargetFiles = glob.glob( f'{targetDir}/val/*' )

        vals_s = create_trainlist( valsourceFiles, scaling_type )
        vals_t = create_trainlist( valtargetFiles, scaling_type )

        val_s = []
        val_t = []
        for l in vals_s:
            l[0] = mirror_padding( l[0], 544, 704 )
            l[1] = mirror_padding( l[1], 544, 704 )
            sepaList = cutting_img( l, 272, 352 )
            val_s.extend( sepaList )
        for l in vals_t:
            l[0] = mirror_padding( l[0], 544, 704 )
            l[1] = mirror_padding( l[1], 544, 704 )
            sepaList = cutting_img( l, 272, 352 )
            val_t.extend( sepaList )

        len_val_s = len( val_s )
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
        
    else:
        if source == 'HeLa':
            train_s = ids_s['train']
            train_t = ids_t['train']
            
            val_s = ids_s['val']
            val_t = ids_t['val']

            len_val_s = len(val_s)
            len_val_t = len(val_t)

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
        if large_flag:
            train_s = []
            train_t = []

            for train_img_list in tmp_train_s:
                train_s.append( random_cropping( train_img_list[0], train_img_list[1], size, size ) )
            for train_img_list in tmp_train_t:
                train_t.append( random_cropping( train_img_list[0], train_img_list[1], size, size ) )

        if source == 'bt474' or 'shsy5y':
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

            #kldiv = nn.KLDivLoss(reduction="batchmean")
            #kldiv(Q.log(), P)
            
            if skipA == False:
                ### process A ( aux update ) ###
                # grad reset
                opt_aux.zero_grad()

                with torch.no_grad():
                    mask_pred_main = net_main(img_s)
                    mask_prob_main = torch.sigmoid(mask_pred_main)
                    mask_prob_flat_main = mask_prob_main.view(-1)

                mask_pred_aux = net_aux(img_s)
                mask_prob_aux = torch.sigmoid(mask_pred_aux)
                mask_prob_flat_aux = mask_prob_aux.view(-1)

                # main, aux のsourceに関するkldiv
                loss_s = kldiv( F.log_softmax(mask_prob_flat_aux), mask_prob_flat_main )
                
                s_epoch_loss += loss_s.item()
                

                # parameter update
                loss_s.backward()
                opt_aux.step()

            ### processB (s1 and s2 update ) ###
            # grad reset
            opt_aux.zero_grad()
            opt_main.zero_grad()

            loss_seg = 0
            
            with torch.no_grad():
                # ensemble により擬似ラベル作成
                # main_pseudo output
                mask_pred_mainp = net_main_pseudo(img_t)
                mask_prob_mainp = torch.sigmoid(mask_pred_mainp)
                mask_prob_flat_mainp = mask_prob_mainp.view(-1)
                # aux_pseudo output
                mask_pred_auxp = net_aux_pseudo(img_t)
                mask_prob_auxp = torch.sigmoid(mask_pred_auxp)
                mask_prob_flat_auxp = mask_prob_auxp.view(-1)

                pseudo_lab = create_ensemble_pseudo( mask_prob_flat_mainp, mask_prob_flat_auxp, device=device )
                
            
            mask_pred_main = net_main(img_t)
            mask_prob_main = torch.sigmoid(mask_pred_main)
            mask_prob_flat_main = mask_prob_main.view(-1)

            mask_pred_aux = net_main(img_t)
            mask_prob_aux = torch.sigmoid(mask_pred_main)
            mask_prob_flat_aux = mask_prob_main.view(-1)


            loss_seg += criterion(mask_prob_flat_aux, pseudo_lab.detach())
            loss_seg += criterion(mask_prob_flat_main, pseudo_lab.detach())

            loss_kl = kldiv( F.log_softmax(mask_prob_flat_aux), mask_prob_flat_main )

            loss = loss_seg / torch.exp( -loss_kl ) + loss_kl
            
            loss.backward()
            opt_main.step()
            opt_aux.step()

            #record discrepancy loss
            d_epoch_loss += loss.item()
            
            
            count += 1
            if (i+1)%10 == 0:
                s_epoch_loss_iter = s_epoch_loss / i
                d_epoch_loss_iter = d_epoch_loss / i
                print('epoch : {}, iter : {}, sourcekl_loss : {}, loss :{}'.format( epoch+1, i, s_epoch_loss_iter, d_epoch_loss_iter ))
        # epoch loss (seg & dis)
        seg = s_epoch_loss / (len_train/batch_size)
        dis = d_epoch_loss / (len_train/batch_size)    
        pseudo_epoch_loss = pseudo_loss / (len_train/batch_size)
        assignedSum_epoch = assignedSum / (len_train/batch_size)
        
        with open(path_w, mode='a') as f:
            f.write('epoch {}: seg:{}, dis:{} \n'.format(epoch + 1, seg, dis))
            
        tr_s_loss_list.append(seg)
        tr_d_loss_list.append(dis)
        pseudo_loss_list.append(pseudo_epoch_loss)
        assigned_list.append(assignedSum_epoch)
        
        
        #s_best_g = net_g.state_dict()
        #s_best_s = net_s1.state_dict()
        #torch.save(s_best_g, '{}CP_G_epoch{}.pth'.format(dir_checkpoint, epoch+1))
        #torch.save(s_best_s, '{}CP_S_epoch{}.pth'.format(dir_checkpoint, epoch+1))
        
        # minimum s_loss or d_loss 更新時checkpoint saved 
        already = False
        if dis < min_val_s_loss:
            min_val_s_loss = dis
            #s_best_g = net_g.state_dict()
            #s_best_s = net_s1.state_dict()
            s_bestepoch = epoch + 1
            #torch.save(s_best_g, '{}CP_G_epoch{}.pth'.format(dir_checkpoint, epoch+1))
            #torch.save(s_best_s, '{}CP_S_epoch{}.pth'.format(dir_checkpoint, epoch+1))
            if saEpoch == None:
                best_main = net_main.state_dict()
                best_aux = net_aux.state_dict()
                
                torch.save({
                    'best_main' : best_main,
                    'best_aux' : best_aux,
                }, '{}CP_min_segloss_e{}'.format(dir_checkpoint, epoch+1))

                already = True
            
            
            with open(path_w, mode='a') as f:
                f.write('loss is update \n')

        
        if ( saEpoch != None ) and ( epoch < saEpoch ):

            best_main = net_main.state_dict()
            best_aux = net_aux.state_dict()
                
            torch.save({
                'best_main' : best_main,
                'best_aux' : best_aux,
            }, '{}CP_min_segloss_e{}'.format(dir_checkpoint, epoch+1))

                
        my_dict = { 'tr_s_loss_list': tr_s_loss_list, 'val_s_loss_list': val_s_loss_list, 'tr_d_loss_list': tr_d_loss_list, 'val_d_loss_list': val_d_loss_list, 'pseudo_loss_list': pseudo_loss_list, 'assigned_list': assigned_list }

        with open(path_lossList, "wb") as tf:
            pickle.dump( my_dict, tf )
                
    
    #segmentation loss graph
    draw_graph( dir_graphs, 'source_loss', epochs, blue_list=tr_s_loss_list, blue_label='train' )

    #discrepancy loss graph
    draw_graph( dir_graphs, 'loss', epochs, red_list=tr_d_loss_list, red_label='train')
    
    """
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
    parser.add_argument('-contmain', '--continue-main', type=str, nargs='?', default=None,
                        help='load checkpoint path?', dest='contrain_main')
    parser.add_argument('-contaux', '--continue-aux', type=str, nargs='?', default=None,
                        help='load checkpoint path?', dest='contrain_aux')
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
    

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    net_main = UNet(first_num_of_kernels=64, n_channels=1, n_classes=1, bilinear=True)
    net_aux = UNet(first_num_of_kernels=32, n_channels=1, n_classes=1, bilinear=True)
    net_main_pseudo = UNet(first_num_of_kernels=64, n_channels=1, n_classes=1, bilinear=True)
    net_aux_pseudo = UNet(first_num_of_kernels=32, n_channels=1, n_classes=1, bilinear=True)
    net_main.to(device=device)
    net_aux.to(device=device)
    net_main_pseudo.to(device=device)
    net_aux_pseudo.to(device=device)

    
    
    checkPoint_main = torch.load(args.contrain_main)
    net_main.load_state_dict(checkPoint_main['best_net'])
    net_main_pseudo.load_state_dict(checkPoint_main['best_net'])
    checkPoint_aux = torch.load(args.contrain_aux)
    net_aux.load_state_dict(checkPoint_aux['best_net'])
    net_aux_pseudo.load_state_dict(checkPoint_aux['best_net'])

    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    opt_main = optim.Adam(
        net_main.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    opt_aux = optim.Adam(
        net_aux.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    opt_main.load_state_dict(checkPoint_main['best_opt'])
    opt_aux.load_state_dict(checkPoint_aux['best_opt'])
    
    for state in opt_main.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    for state in opt_aux.state.values():
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
        
        train_net(net_main=net_main,
                  net_aux=net_aux,
                  net_main_pseudo=net_main_pseudo,
                  net_aux_pseudo=net_aux_pseudo,
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
                  opt_main=opt_main,
                  opt_aux=opt_aux,
                  skipA=args.skipA,
                  Bssl=args.Bssl,
                  pseConf=args.pseConf)
        
            
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
