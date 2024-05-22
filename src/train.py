import sys
import pickle
import argparse
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import torch.nn.functional as F
from model import *
from functions_io import *


def get_args():
    parser = argparse.ArgumentParser(description='Train model with input images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset-dir', metavar='D', type=str, default='./dataset',
                        help='Dataset directory path', dest='dataset_dir')
    parser.add_argument('-o', '--output-dir', metavar='O', type=str, nargs='?', default='./result/train',
                        help='output directory?', dest='output_dir')
    parser.add_argument('-s', '--source', metavar='S', type=str, nargs='?', default='shsy5y',
                        help='source cell', dest='source')
    parser.add_argument('-t', '--target', metavar='T', type=str, nargs='?', default='mcf7',
                        help='target cell', dest='target')
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
                        help='First num of kernels', dest='first_num_of_kernels')
    parser.add_argument('-conf', '--pse-conf', metavar='PSEC', type=float, nargs='?', default=1.0,
                        help='the confidence of pseudo label?', dest='c_conf')
    parser.add_argument('-scaling', '--scaling-type', metavar='SM', type=str, nargs='?', default='normal',
                        help='scaling method??', dest='scaling_type')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=3,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-om', '--optimizer-method', metavar='OM', type=str, nargs='?', default='Adam',
                        help='Optimizer method', dest='optimizer_method')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu number?', dest='gpu_no')
    parser.add_argument('-seed', type=int, nargs='?', default=0,
                        help='seed num?', dest='seed')
    parser.add_argument('-c', '--checkpoint', metavar='CT', type=str, nargs='?', default=None,
                        help='load checkpoint path?', dest='checkpoint')

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


def train_net(net_1,
              net_2,
              device,
              epochs=5,
              batch_size=4,
              lr=0.0001,
              first_num_of_kernels=64,
              dir_datasets='./datasets',
              dir_checkpoint='./checkpoint',
              dir_result='./result',
              dir_graphs='./result/graphs',
              optimizer_method='Adam',
              source='HeLa',
              target='3T3',
              scaling_type='normal',
              opt_1=None,
              opt_2=None,
              c_conf=1):

    # resultfile & losslist
    path_w = f"{dir_result}/output.txt"
    path_lossList = f"{dir_graphs}/loss_list.pkl"
    
    # recode training conditions
    with open( path_w, mode='w' ) as f:  
        f.write( 'first num of kernels:{} \n'.format( first_num_of_kernels ) )
        f.write( 'optimizer method:{}, learning rate:{} \n'.format( optimizer_method, lr ) )
        f.write( 'max epoch:{}, batchsize:{} \n'.format( epochs, batch_size ) )
        f.write( f'src:{source}, tgt:{target}\n' )
        f.write( f'c_conf:{c_conf}  \n' )
        
    # optimizer set
    criterion = nn.BCELoss()

    # set dataset dir
    sourceDir = f'{dir_datasets}/train_data/{source}'
    targetDir = f'{dir_datasets}/train_data/{target}'

    # load train images
    trsourceFiles = glob.glob(f'{sourceDir}/train/*')

    ## cat_train = train & val
    trtargetFiles = glob.glob(f'{targetDir}/cat_train/*')

    trains_s = create_trainlist( trsourceFiles, scaling_type )
    trains_t = create_trainlist( trtargetFiles, scaling_type )

    # train: (520, 704)->(560, 784)
    for k in trains_s:
        k[0] = mirror_padding( k[0], 560, 784 )
        k[1] = mirror_padding( k[1], 560, 784 )
    for k in trains_t:
        k[0] = mirror_padding( k[0], 560, 784 )
        k[1] = mirror_padding( k[1], 560, 784 )
    # adjust len(train_s) == len(train_t)
    if len( trains_s ) > len( trains_t ):
        diff_s_t = len( trains_s ) - len( trains_t )
        len_train = len( trains_s )
        source_extend = False
    else:
        diff_s_t = len( trains_t ) - len( trains_s )
        len_train = len( trains_t )
        source_extend = True
        
            
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
    tr_loss_list = []
    source_seg_list_1 = []
    source_seg_list_2 = []
    tgt_seg_list_1 = []
    tgt_seg_list_2 = []
    jsdiv_loss_list = []
    val_iou_t1_list = []
    val_iou_t2_list = []
    min_val_s_loss_1 = 10000.0

    len_trs = len(trains_s)
    
    for epoch in range(epochs):
        
        count = 0
        train_s = []
        train_t = []

        # adjust num of source & target data
        if source_extend:
            if diff_s_t > len_trs:
                diff_s_t_2 = diff_s_t - len_trs
                tmps = trains_s.copy()
                trains_s.extend( tmps )
                trains_s.extend( random.sample( trains_s, diff_s_t_2 ) )
            else:
                trains_s.extend( random.sample( trains_s, diff_s_t ) )
        # random cropping source from mirror pad. img
        for train_img_list in trains_s:
            train_s.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        random.shuffle( train_s )
        
        # adjust num of source & target data
        if not source_extend:
            trains_t.extend( random.sample( trains_t, diff_s_t ) )
        # random cropping target from mirror pad. img
        for train_img_list in trains_t:
            train_t.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        random.shuffle( train_t )
        
        #---- Train section
        epoch_loss = 0
        source_seg_epoch_loss_1 = 0
        source_seg_epoch_loss_2 = 0
        tgt_seg_epoch_loss_1 = 0
        tgt_seg_epoch_loss_2 = 0

        for i, (bs, bt) in enumerate(zip(batch(train_s, batch_size, source), batch(train_t, batch_size, source))):

            # source data
            img_s = np.array([i[0] for i in bs]).astype(np.float32)
            img_s = torch.from_numpy(img_s).to(device=device)
            # source label
            mask = np.array([i[1] for i in bs]).astype(np.float32)
            mask = torch.from_numpy(mask).to(device=device)
            mask_flat = mask.view(-1)

            # target data
            img_t = np.array([i[0] for i in bt]).astype(np.float32)
            img_t = torch.from_numpy(img_t).to(device=device)


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

            pseudo_lab_t1, pseudo_lab_t2, confidence = create_uncer_pseudo( mask_prob_1_flat, mask_prob_2_flat, device=device )

            confidence = c_conf * confidence

            pseudo_criterion = nn.BCELoss( weight=confidence.detach() )

            # crossing field
            t_loss_1 = pseudo_criterion( mask_prob_1_flat, pseudo_lab_t2.detach() )
            t_loss_2 = pseudo_criterion( mask_prob_2_flat, pseudo_lab_t1.detach() )

            loss_total = s_loss_1 + s_loss_2 + t_loss_1 + t_loss_2

            opt_1.zero_grad()
            opt_2.zero_grad()

            loss_total.backward()

            # net1 & net2 update!
            opt_1.step()
            opt_2.step()

            source_seg_epoch_loss_1 += s_loss_1.item()
            source_seg_epoch_loss_2 += s_loss_2.item()

            tgt_seg_epoch_loss_1 += t_loss_1.item()
            tgt_seg_epoch_loss_2 += t_loss_2.item()

            epoch_loss += loss_total.item()
            count += 1
            if i != 0:
                print('epoch : {}, iter : {}, loss : {}'.format( epoch+1, i, epoch_loss/i ) )

        ### finish training
        seg = epoch_loss / (len_train/batch_size)
        
        with open(path_w, mode='a') as f:
            f.write('epoch {}: seg:{} \n'.format(epoch + 1, seg))

        tr_loss_list.append(seg)

        source_seg_list_1.append(source_seg_epoch_loss_1/(len_train/batch_size))
        source_seg_list_2.append(source_seg_epoch_loss_2/(len_train/batch_size))
        
        tgt_seg_list_1.append(tgt_seg_epoch_loss_1/(len_train/batch_size))
        tgt_seg_list_2.append(tgt_seg_epoch_loss_2/(len_train/batch_size))

        #---- Val section
        ####################################################################################
        val_iou_t1 = 0
        val_iou_t2 = 0
        with torch.no_grad():

            for k, bt in enumerate(val_t):
                ###
                img_t = np.array(bt[0]).astype(np.float32)
                img_t = img_t.reshape([1, img_t.shape[-2], img_t.shape[-1]])
                img_t =  torch.from_numpy(img_t).unsqueeze(0).to(device=device)
                ###
                mask = np.array(bt[1]).astype(np.float32)
                mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                mask = torch.from_numpy(mask).unsqueeze(0).to(device=device)
                mask_flat = mask.view(-1)
                ####
                
                #s1,s2 output----------------
                mask_pred_1 = net_1(img_t)
                mask_pred_2 = net_2(img_t)

                mask_prob_1 = torch.sigmoid(mask_pred_1)
                mask_prob_2 = torch.sigmoid(mask_pred_2)
                #----------------

                
                mask_bin_1 = (mask_prob_1[0] > 0.5).float()
                mask_bin_2 = (mask_prob_2[0] > 0.5).float()
                val_iou_t1 += iou_loss(mask_bin_1, mask.float(), device).item()
                val_iou_t2 += iou_loss(mask_bin_2, mask.float(), device).item()
        
        #valdice_list.append(val_dice / len_val_s)
        val_iou_t1_list.append( val_iou_t1 / len_val_t )
        val_iou_t2_list.append( val_iou_t2 / len_val_t )
        ####################################################################################
        if seg < min_val_s_loss_1:
            min_val_s_loss_1 = seg

            best_net1 = net_1.state_dict()
            best_net2 = net_2.state_dict()

            best_opt1 = opt_1.state_dict()
            best_opt2 = opt_2.state_dict()

            torch.save({
                'best_net1': best_net1,
                'best_net2': best_net2,
                'opt_net1': best_opt1,
                'opt_net2': best_opt2,
                'epoch': epoch + 1,
            }, f'{dir_checkpoint}/best_learned_model')

            with open(path_w, mode='a') as f:
                f.write('best model was saved \n')
                
        my_dict = { 'tr_loss_list': tr_loss_list,
                    'source_seg_list_1': source_seg_list_1,
                    'source_seg_list_2': source_seg_list_2,
                    'tgt_seg_list_1':tgt_seg_list_1,
                    'tgt_seg_list_2':tgt_seg_list_2,
                    'jsdiv_loss_list':jsdiv_loss_list,
                    'val_iou_t1_list':val_iou_t1_list,
                    'val_iou_t2_list':val_iou_t2_list
                    }

        with open(path_lossList, "wb") as tf:
            pickle.dump(my_dict, tf)
                
    
    #segmentation loss graph
    
    draw_graph( dir_graphs, 'train_loss', epochs, red_list=tr_loss_list,  red_label='loss')

    draw_graph( dir_graphs, 'loss_net1', epochs, red_list=source_seg_list_1,  red_label='source loss', green_list=tgt_seg_list_1, green_label='target loss' )

    draw_graph( dir_graphs, 'loss_net2', epochs, red_list=source_seg_list_2,  red_label='source loss', green_list=tgt_seg_list_2, green_label='target loss' )

    draw_graph( dir_graphs, 'target_IoU', epochs, red_list=val_iou_t1_list,  red_label='net1_IoU', green_list=val_iou_t2_list,  green_label='net2_IoU', y_label='IoU' )


if __name__ == '__main__':
    args = get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_no}'
    device = torch.device(f'cuda:{args.gpu_no}' if torch.cuda.is_available() else 'cpu')

    # fix the seed
    torch_fix_seed(args.seed)
    
    # co-teaching two networks
    net_1 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_1.to(device=device)
    net_2 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_2.to(device=device)

    if args.optimizer_method == 'Adam':
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
    elif args.optimizer_method == 'SGD':
        opt_1 = optim.SGD(
            net_1.parameters(),
            lr=args.lr,
            weight_decay=0,
        )

        opt_2 = optim.SGD(
            net_2.parameters(),
            lr=args.lr,
            weight_decay=0,
        )
    else:
        raise NotImplementedError
         
    
    # loading models from checkpoints
    if args.checkpoint is not None:
        checkPoint = torch.load(args.checkpoint, map_location=device)
        net_1.load_state_dict(checkPoint['best_net1'])
        net_2.load_state_dict(checkPoint['best_net2'])
        opt_1.load_state_dict(checkPoint['best_opt1'])
        opt_2.load_state_dict(checkPoint['best_opt2'])
            
    
    # optimizer to cuda
    for state in opt_1.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)

    for state in opt_2.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    
    dir_datasets = f'{args.dataset_dir}'
    dir_result = f'{args.output_dir}'
    dir_checkpoint = f'{dir_result}/checkpoint'
    dir_graphs = f'{dir_result}/graphs'
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(dir_graphs, exist_ok=True)
    
    try:
        
        train_net(net_1=net_1,
                  net_2=net_2,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  first_num_of_kernels=args.first_num_of_kernels,
                  device=device,
                  dir_datasets=f'{dir_datasets}',
                  dir_checkpoint=f'{dir_checkpoint}',
                  dir_result=f'{dir_result}',
                  dir_graphs=f'{dir_graphs}',
                  optimizer_method=args.optimizer_method,
                  source=args.source,
                  target=args.target,
                  scaling_type=args.scaling_type,
                  opt_1=opt_1,
                  opt_2=opt_2,
                  c_conf=args.c_conf)
        
            
    except KeyboardInterrupt:
        #torch.save(net_.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
