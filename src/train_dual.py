import sys
import time
import pickle
import argparse
import torch
import torch.nn as nn
from torch import optim
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from model import *
from functions_io import *
from core.dual.augmentations import ClassMixLoss, compute_classmix, compute_cutmix, compute_ic
from core.dual.optimizer import PolyWarmupAdamW
from core.dual.model import MiT_SegFormer
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='Train model with input images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset-dir', metavar='D', type=str, default='./LIVECell_dataset',
                        help='Dataset directory path', dest='dataset_dir')
    parser.add_argument('-o', '--output-dir', metavar='O', type=str, nargs='?', default='./result/train',
                        help='output directory?', dest='output_dir')
    parser.add_argument('-s', '--source', metavar='S', type=str, nargs='?', default='shsy5y',
                        help='source cell', dest='source')
    parser.add_argument('-t', '--target', metavar='T', type=str, nargs='?', default='mcf7',
                        help='target cell', dest='target')
    parser.add_argument('-scaling', '--scaling-type', metavar='SM', type=str, nargs='?', default='unet',
                        help='scaling method??', dest='scaling_type')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=400,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
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
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
                        help='First num of kernels', dest='first_num_of_kernels')

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


def update_ema(model_teacher, model, alpha_teacher, iteration):
    with torch.no_grad():
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
        for ema_param, param in zip(model_teacher.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]


def train_net(model=None,
              model_teacher=None,
              model_teacher2=None,
              optimizer=None,
              device=None,
              epochs=5,
              batch_size=4,
              lr=0.0001,
              dir_datasets='./datasets',
              dir_checkpoint='./checkpoint',
              dir_graphs='./result/graphs',
              source='HeLa',
              target='3T3',
              scaling_type='normal',
              num_classes=2,
              ):

    # resultfile & losslist
    path_lossList = f"{dir_graphs}/loss_list.pkl"
        
    # optimizer set
    criterion = torch.nn.CrossEntropyLoss().to(device=device)
    criterion_u = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)
    cm_loss_fn = ClassMixLoss(weight=None, reduction='none', ignore_index=255)

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
        
    # adjust num of source & target data
    len_trs = len(trains_s)
    if source_extend:
        if diff_s_t > len_trs:
            diff_s_t_2 = diff_s_t - len_trs
            tmps = trains_s.copy()
            trains_s.extend( tmps )
            trains_s.extend( random.sample( trains_s, diff_s_t_2 ) )
        else:
            trains_s.extend( random.sample( trains_s, diff_s_t ) )
    # adjust num of source & target data
    if not source_extend:
        trains_t.extend( random.sample( trains_t, diff_s_t ) )
    # load val images
    valtargetFiles = glob.glob( f'{targetDir}/val/*' )

    vals_t = create_trainlist( valtargetFiles, scaling_type )
    val_t = []

    for l in vals_t:
        l[0] = mirror_padding( l[0], 544, 704 )
        l[1] = mirror_padding( l[1], 544, 704 )
        sepaList = cutting_img( l, 272, 352 )
        val_t.extend(sepaList)

    len_val_t = len(val_t)
    val_iou_mean_best = 0.0

    total_loss_dict = {epoch: [] for epoch in range(epochs)}
    val_iou_dict = {epoch: [] for epoch in range(epochs)}

    
    for epoch in range(epochs):

        start = time.time()

        model.train()
        model_teacher.train()
        model_teacher2.train()

        if epoch % 2 == 0:
            ema_model = model_teacher
            do_cut_mix = True
            do_class_mix = False
        else:
            ema_model = model_teacher2
            do_cut_mix = False
            do_class_mix = True

        ema_model.train()

        count = 0
        train_s = []
        train_t = []

        # random cropping source from mirror pad. img
        for train_img_list in trains_s:
            train_s.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        random.shuffle(train_s)

        # random cropping target from mirror pad. img
        for train_img_list in trains_t:
            train_t.append( random_cropping( train_img_list[0], train_img_list[1], 272, 352 ) )
        random.shuffle(train_t)
        
        #---- Train section
        iteration = 0

        for i, (bs, bt) in enumerate(zip(batch(train_s, batch_size, source), batch(train_t, batch_size, source))):
            # source data and label
            i_s_list = []
            l_s_list = []
            for i_s, l_s in bs:
                i_s_list.append(i_s)
                if num_classes != 1:
                    l_s = np.eye(num_classes)[l_s.astype(int)]  # One-Hot Encoding
                    l_s = l_s[0].transpose(2, 0, 1)
                l_s_list.append(l_s)

            img_s = np.array(i_s_list).astype(np.float32)
            img_s = torch.from_numpy(img_s).to(device=device)
            mask_s = np.array(l_s_list).astype(np.float32)
            mask_s = torch.from_numpy(mask_s).to(device=device)

            # target data and label
            i_t_list = []
            l_t_list = []
            for i_t, l_t in bt:
                i_t_list.append(i_t)
                if num_classes != 1:
                    l_t = np.eye(num_classes)[l_t.astype(int)]  # One-Hot Encoding
                    l_t = l_t[0].transpose(2, 0, 1)
                l_t_list.append(l_t)

            img_t = np.array(i_t_list).astype(np.float32)
            img_t = torch.from_numpy(img_t).to(device=device)
            mask_t = np.array(l_t_list).astype(np.float32)
            mask_t = torch.from_numpy(mask_t).to(device=device)

            # assign each data to adjust original variables in their codes
            image = img_s
            label = mask_s
            image_u = img_t
            label_u = mask_t
            image_u_strong = deepcopy(image_u)
            image_u_strong = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_u_strong)
            image_u_strong = transforms.RandomGrayscale(p=0.2)(image_u_strong)

            # loss
            assert img_s.shape == img_t.shape, f'Image shape was not matched. source: {img_s.shape}, target: {img_t.shape}'
            b, _, h, w = img_s.shape

            loss = None
            if do_class_mix:
                loss = compute_classmix(device, b, h, w, criterion, cm_loss_fn, model, ema_model, image, label, image_u,
                                        image_u_strong, threshold=0.95)
            if do_cut_mix:
                loss = compute_cutmix(device, h, w, image, label, criterion, model, ema_model, image_u, threshold=0.95)

            loss_dc = compute_ic(model, ema_model, image_u, image_u_strong, criterion_u, label_u, h, w, threshold=0.95)
            total_loss = loss + loss_dc * 0.2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema(model_teacher=ema_model, model=model, alpha_teacher=0.99, iteration=i)

            iteration += 1

            total_loss_dict[epoch].append(total_loss.detach().clone().cpu().numpy())

        total_loss_mean = float(abs(np.mean(total_loss_dict[epoch])))
        print(f'epoch: {epoch+1}, total_loss: {total_loss_mean}')

        #---- Val section
        model.eval()

        with torch.no_grad():
            for k, bt in enumerate(val_t):
                ###
                img_t = np.array(bt[0]).astype(np.float32)
                if len(img_t.shape) == 2:
                    img_t = np.expand_dims(img_t, 2)
                img_t = np.transpose(img_t, (2, 0, 1))
                img_t = torch.from_numpy(img_t).unsqueeze(0).to(device=device)
                ###
                mask = np.array(bt[1]).astype(np.float32)
                mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                mask = torch.from_numpy(mask).unsqueeze(0).to(device=device)
                # mask = np.array(bt[1]).astype(np.float32)
                # mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
                # if num_classes != 1:
                #     mask = np.eye(num_classes)[mask.astype(int)]  # One-Hot Encoding
                #     mask = mask[0].transpose(2, 0, 1)
                # mask = torch.from_numpy(mask).unsqueeze(0).to(device=device)
                ####
                
                # infer
                output = model(img_t)
                #output = F.interpolate(output, size=mask.shape[1:], mode='bilinear', align_corners=False)
                output = F.interpolate(output, size=mask.shape[2:], mode='bilinear', align_corners=False)
                output = torch.argmax(output, dim=1)

                val_iou = iou_loss(output, mask.float(), device).item()
                val_iou_dict[epoch].append(val_iou)

        val_iou_mean = float(abs(np.mean(val_iou_dict[epoch])))
        print(f'val_iou: {val_iou_mean}')

        my_dict = {
            'total_loss_dict': total_loss_dict,
            'val_iou_dict': val_iou_dict
        }
        with open(path_lossList, "wb") as tf:
            pickle.dump(my_dict, tf)
        ####################################################################################
        if val_iou_mean > val_iou_mean_best:
            val_iou_mean_best = val_iou_mean

            best_model = model.state_dict()
            best_model_teacher = model_teacher.state_dict()
            best_model_teacher2 = model_teacher2.state_dict()

            best_optimizer = optimizer.state_dict()

            torch.save({
                'best_model': best_model,
                'best_model_teacher': best_model_teacher,
                'best_model_teacher2': best_model_teacher2,
                'best_optimizer': best_optimizer,
                'epoch': epoch + 1,
            }, f'{dir_checkpoint}/best_learned_model')
        torch.cuda.empty_cache()
        elapsed_time = time.time() - start
        print(f'Time: {elapsed_time}')


if __name__ == '__main__':
    args = get_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_no}'
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # fix the seed
    torch_fix_seed(args.seed)
    
    # build models
    num_classes = 2
    # backbone = 'mit_b1'
    # model = MiT_SegFormer(backbone=backbone,
    #                       num_classes=num_classes,
    #                       embedding_dim=256,
    #                       pretrained=False)
    # model_teacher = MiT_SegFormer(backbone=backbone + '_ema',
    #                               num_classes=num_classes,
    #                               embedding_dim=256,
    #                               pretrained=False)
    # model_teacher2 = MiT_SegFormer(backbone=backbone + '_ema',
    #                                num_classes=num_classes,
    #                                embedding_dim=256,
    #                                pretrained=False)
    model = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=num_classes, bilinear=True)
    model_teacher = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=num_classes, bilinear=True)
    model_teacher2 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=num_classes, bilinear=True)

    model.to(device)
    model_teacher.to(device)
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher2.to(device)
    for p in model_teacher2.parameters():
        p.requires_grad = False

    # param_groups = model.get_param_groups()
    #
    # optimizer = PolyWarmupAdamW(
    #     params=[
    #         {
    #             "params": param_groups[0],
    #             "lr": args.lr,
    #             "weight_decay": 0.01,
    #         },
    #         {
    #             "params": param_groups[1],
    #             "lr": args.lr,
    #             "weight_decay": 0.0,
    #         },
    #         {
    #             "params": param_groups[2],
    #             "lr": args.lr * 10,
    #             "weight_decay": 0.01,
    #         },
    #     ],
    #     lr=args.lr,
    #     weight_decay=0.01,
    #     betas=(0.9, 0.999),
    #     warmup_iter=16,  # 1500/50000 → 16/527
    #     max_iter=527,
    #     warmup_ratio=1e-6,
    #     power=1.0
    # )

    if args.optimizer_method == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )
    elif args.optimizer_method == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=0,
        )
    else:
        raise NotImplementedError

    # optimizer to cuda
    optimizer_to_cuda(optimizer, device)

    # set paths
    dir_datasets = f'{args.dataset_dir}'
    dir_result = f'{args.output_dir}'
    print(f'Save path: {dir_result}')
    dir_checkpoint = f'{dir_result}/checkpoint'
    dir_graphs = f'{dir_result}/graphs'
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(dir_graphs, exist_ok=True)
    
    # train
    train_net(model=model,
              model_teacher=model_teacher,
              model_teacher2=model_teacher2,
              optimizer=optimizer,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              dir_datasets=f'{dir_datasets}',
              dir_checkpoint=f'{dir_checkpoint}',
              dir_graphs=f'{dir_graphs}',
              source=args.source,
              target=args.target,
              scaling_type=args.scaling_type,
              num_classes=num_classes,
              )
