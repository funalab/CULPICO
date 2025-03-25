import os.path
import sys
import json
import pickle
import argparse
import time
import gc
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import torch.nn.functional as F
from model import *
from functions_io import *
from core.cra.build import resnet_feature_extractor, ASPP_Classifier_V2, PixelDiscriminator, adjust_learning_rate
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


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    loss = -soft_label.float()*F.log_softmax(pred, dim=1) #F.logsigmoid(pred)#
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))


def train_net(feature_extractor=None,
              classifier=None,
              model_D=None,
              optimizer_fea=None,
              optimizer_cls=None,
              optimizer_D=None,
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
              last_epoch=0,
              ):

    # resultfile & losslist
    path_lossList = f"{dir_graphs}/loss_list.pkl"
        
    # optimizer set
    criterion = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss(reduction='none')

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
            trains_s.extend(trains_s.copy())
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
        sepaList = cutting_img_for_resnet( l, 272, 352 )
        val_t.extend(sepaList)
    del vals_t

    len_val_t = len(val_t)
    len_trs = len(trains_s)
    val_iou_mean_best = 0.0

    loss_seg_dict = {epoch: [] for epoch in range(epochs)}
    loss_adv_tgt_dict = {epoch: [] for epoch in range(epochs)}
    loss_D_src_dict = {epoch: [] for epoch in range(epochs)}
    loss_D_tgt_dict = {epoch: [] for epoch in range(epochs)}
    val_iou_dict = {epoch: [] for epoch in range(epochs)}

    if last_epoch != 0 and os.path.exists(path_lossList):
        with open(path_lossList,'rb') as f:
            last_data = pickle.load(f)
        loss_seg_dict = last_data['loss_seg_dict']
        loss_adv_tgt_dict = last_data['loss_adv_tgt_dict']
        loss_D_src_dict = last_data['loss_D_src_dict']
        loss_D_tgt_dict = last_data['loss_D_tgt_dict']
        val_iou_dict = last_data['val_iou_dict']

        with open(os.path.join(dir_checkpoint, f'last_epoch.json'), 'r') as f:
            last_log = json.load(f)
            val_iou_mean_best = last_log['val_iou_mean_best']

    for epoch in range(last_epoch, epochs):

        start = time.time()

        feature_extractor.train()
        classifier.train()
        model_D.train()

        count = 0
        train_s = []
        train_t = []

        # random cropping source from mirror pad. img
        for train_img_list in trains_s:
            train_s.append( random_cropping_for_resnet( train_img_list[0], train_img_list[1], 272, 352 ) )
        random.shuffle(train_s)

        # random cropping target from mirror pad. img
        for train_img_list in trains_t:
            train_t.append( random_cropping_for_resnet( train_img_list[0], train_img_list[1], 272, 352 ) )
        random.shuffle(train_t)
        
        #---- Train section
        iteration = 0
        max_iters = len(list(batch_for_resnet(train_s, batch_size)))

        for i, (bs, bt) in enumerate(zip(batch_for_resnet(train_s, batch_size), batch_for_resnet(train_t, batch_size))):
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
            mask = np.array(l_s_list).astype(np.float32)
            mask = torch.from_numpy(mask).to(device=device)

            # target data
            img_t = np.array([i[0] for i in bt]).astype(np.float32)
            img_t = torch.from_numpy(img_t).to(device=device)

            # adujest learning rate
            current_lr = adjust_learning_rate(method='poly', base_lr=lr,
                                              iters=iteration, max_iters=max_iters,
                                              power=0.9)
            current_lr_D = adjust_learning_rate(method='poly', base_lr=lr,
                                                iters=iteration, max_iters=max_iters,
                                                power=0.9)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr * 10
            for index in range(len(optimizer_D.param_groups)):
                optimizer_D.param_groups[index]['lr'] = current_lr_D

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_D.zero_grad()
            src_input = img_s.to(device=device, non_blocking=True)
            src_label = mask.to(device=device, non_blocking=True)
            tgt_input = img_t.to(device=device, non_blocking=True)

            src_size = src_input.shape[-2:]
            tgt_size = tgt_input.shape[-2:]

            src_fea = feature_extractor(src_input)
            src_pred = classifier(src_fea, src_size)
            temperature = 1.8
            src_pred = src_pred.div(temperature)
            loss_seg = criterion(src_pred, src_label)
            loss_seg.backward()
            # generate soft labels
            src_soft_label = F.softmax(src_pred, dim=1).detach() #torch.sigmoid(src_pred).detach()
            src_soft_label[src_soft_label > 0.9] = 0.9

            tgt_fea = feature_extractor(tgt_input)
            tgt_pred = classifier(tgt_fea, tgt_size)
            tgt_pred = tgt_pred.div(temperature)
            tgt_soft_label = F.softmax(tgt_pred, dim=1) # torch.sigmoid(tgt_pred)

            tgt_soft_label = tgt_soft_label.detach()
            tgt_soft_label[tgt_soft_label > 0.9] = 0.9

            tgt_D_pred = model_D(tgt_fea, tgt_size)
            loss_adv_tgt = 0.001 * soft_label_cross_entropy(tgt_D_pred, torch.cat(
                (tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
            loss_adv_tgt.backward()

            optimizer_fea.step()
            optimizer_cls.step()

            optimizer_D.zero_grad()

            src_D_pred = model_D(src_fea.detach(), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(src_D_pred,
                                                        torch.cat((src_soft_label, torch.zeros_like(src_soft_label)),
                                                                  dim=1))
            loss_D_src.backward()

            tgt_D_pred = model_D(tgt_fea.detach(), tgt_size)
            loss_D_tgt = 0.5 * soft_label_cross_entropy(tgt_D_pred,
                                                        torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label),
                                                                  dim=1))
            loss_D_tgt.backward()


            optimizer_D.step()

            iteration += 1

            loss_seg_dict[epoch].append(loss_seg.detach().clone().cpu().numpy())
            loss_adv_tgt_dict[epoch].append(loss_adv_tgt.detach().clone().cpu().numpy())
            loss_D_src_dict[epoch].append(loss_D_src.detach().clone().cpu().numpy())
            loss_D_tgt_dict[epoch].append(loss_D_tgt.detach().clone().cpu().numpy())

        del train_s, train_t
        gc.collect()
        loss_seg_mean = float(abs(np.mean(loss_seg_dict[epoch])))
        loss_adv_tgt_mean = float(abs(np.mean(loss_adv_tgt_dict[epoch])))
        loss_D_src_mean = float(abs(np.mean(loss_D_src_dict[epoch])))
        loss_D_tgt_mean = float(abs(np.mean(loss_D_tgt_dict[epoch])))
        print(f'epoch: {epoch+1}, loss_seg: {loss_seg_mean}, loss_adv_tgt: {loss_adv_tgt_mean},'
              f'loss_D_src: {loss_D_src_mean}, loss_D_tgt: {loss_D_tgt_mean}')

        #---- Val section
        feature_extractor.eval()
        classifier.eval()

        with torch.no_grad():
            for k, bt in enumerate(val_t):
                ###
                img_t = np.array(bt[0]).astype(np.float32)
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
                size = mask.shape[-2:]
                output = classifier(feature_extractor(img_t), size)
                output = output.max(1)[1]

                val_iou = iou_loss(output, mask.float(), device).item()
                val_iou_dict[epoch].append(val_iou)

        val_iou_mean = float(abs(np.mean(val_iou_dict[epoch])))
        print(f'val_iou: {val_iou_mean}')

        my_dict = {
            'loss_seg_dict': loss_seg_dict,
            'loss_adv_tgt_dict': loss_adv_tgt_dict,
            'loss_D_src_dict': loss_D_src_dict,
            'loss_D_tgt_dict': loss_D_tgt_dict,
            'val_iou_dict': val_iou_dict
        }
        with open(path_lossList, "wb") as tf:
            pickle.dump(my_dict, tf)

        # save final model
        last_ckpt = {
            'epoch': epoch + 1,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(legacy=False),
            'torch_random_state': torch.random.get_rng_state(),
            'feature_extractor': feature_extractor.state_dict(),
            'classifier': classifier.state_dict(),
            'model_D': model_D.state_dict(),
            'optimizer_fea': optimizer_fea.state_dict(),
            'optimizer_cls': optimizer_cls.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }

        if device == torch.device('cuda') or device == torch.device('mps'):
            last_ckpt['torch_cuda_random_state'] = torch.cuda.get_rng_state()

        torch.save(last_ckpt, os.path.join(dir_checkpoint, f'last_epoch_object.cpt'))

        save_data_for_last = dict()
        save_data_for_last['last_epoch'] = epoch + 1
        save_data_for_last['val_iou_mean_best'] = val_iou_mean_best

        with open(os.path.join(dir_checkpoint, f'last_epoch.json'), 'w') as f:
            json.dump(save_data_for_last, f, indent=4)

        ####################################################################################
        if val_iou_mean > val_iou_mean_best:
            val_iou_mean_best = val_iou_mean

            torch.save({
                'best_feature_extractor': feature_extractor.state_dict(),
                'best_classifier': classifier.state_dict(),
                'best_model_D': model_D.state_dict(),
                'best_optimizer_fea': optimizer_fea.state_dict(),
                'best_optimizer_cls': optimizer_cls.state_dict(),
                'best_optimizer_D': optimizer_D.state_dict(),
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

    feature_extractor = resnet_feature_extractor("resnet101",
                                                 pretrained_weights=None,#"https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
                                                 aux=False,
                                                 pretrained_backbone=False,#True,
                                                 freeze_bn=False)
    feature_extractor.to(device)

    num_classes = 2
    classifier = ASPP_Classifier_V2(in_channels=2048, dilation_series=[6, 12, 18, 24], padding_series=[6, 12, 18, 24],
                                    num_classes=num_classes)
    classifier.to(device)

    model_D = PixelDiscriminator(input_nc=2048, ndf=256, num_classes=num_classes)
    model_D.to(device)

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(),
                                    lr=args.lr,
                                    momentum=0,
                                    weight_decay=0)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(),
                                    lr=args.lr * 10,
                                    momentum=0,
                                    weight_decay=0)
    optimizer_cls.zero_grad()

    optimizer_D = torch.optim.Adam(model_D.parameters(),
                                   lr=args.lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=0,
                                   amsgrad=False
                                   )
    optimizer_D.zero_grad()

    # continue train by using checkpoint
    last_epoch = 0
    if args.checkpoint is not None:
        checkPoint = torch.load(args.checkpoint, map_location=device)
        feature_extractor.load_state_dict(checkPoint['feature_extractor'])
        classifier.load_state_dict(checkPoint['classifier'])
        model_D.load_state_dict(checkPoint['model_D'])
        optimizer_fea.load_state_dict(checkPoint['optimizer_fea'])
        optimizer_cls.load_state_dict(checkPoint['optimizer_cls'])
        optimizer_D.load_state_dict(checkPoint['optimizer_D'])

        random.setstate(checkPoint["random_state"])
        np.random.set_state(checkPoint['np_random_state'])
        torch.random.set_rng_state(checkPoint["torch_random_state"])
        if 'torch_cuda_random_state' in checkPoint:
            torch.cuda.set_rng_state(checkPoint['torch_cuda_random_state'])

        with open(os.path.join(os.path.dirname(args.checkpoint), 'last_epoch.json'),'r') as f:
            data = json.load(f)
        last_epoch = data['last_epoch']

    # optimizer to cuda
    optimizer_to_cuda(optimizer_fea, device)
    optimizer_to_cuda(optimizer_cls, device)
    optimizer_to_cuda(optimizer_D, device)

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
    train_net(feature_extractor=feature_extractor,
              classifier=classifier,
              model_D=model_D,
              optimizer_fea=optimizer_fea,
              optimizer_cls=optimizer_cls,
              optimizer_D=optimizer_D,
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
              last_epoch=last_epoch
              )
