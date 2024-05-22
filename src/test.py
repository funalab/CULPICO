import argparse
from functions_io import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import glob
from skimage import io
import statistics
import os


def eval_mcd(device, test_list, model=None, model_2=None,
             logfilePath=None):
    IoU_list = []
    precision_list = []
    recall_list = []

    tf = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(h),
        transforms.ToTensor()
    ])

    for i, image in enumerate(test_list):

        # left, right, up, down
        # input dim = 3
        img_left_up = torch.from_numpy(image[0][0]).unsqueeze(0).cuda(device)
        img_right_up = torch.from_numpy(image[1][0]).unsqueeze(0).cuda(device)
        img_left_down = torch.from_numpy(image[2][0]).unsqueeze(0).cuda(device)
        img_right_down = torch.from_numpy(image[3][0]).unsqueeze(0).cuda(device)
        # output & ground truth dim = 2
        gt_left_up = torch.from_numpy(image[0][1][0]).cuda(device)
        gt_right_up = torch.from_numpy(image[1][1][0]).cuda(device)
        gt_left_down = torch.from_numpy(image[2][1][0]).cuda(device)
        gt_right_down = torch.from_numpy(image[3][1][0]).cuda(device)
        ## gt concat
        gt_left = torch.cat((gt_left_up[0:260, ], gt_left_down[12:, ]), dim=0)
        gt_right = torch.cat((gt_right_up[0:260, ], gt_right_down[12:, ]), dim=0)
        gt = torch.cat((gt_left, gt_right), dim=1)

        with torch.no_grad():
            if model_2 is None:
                mask_lu = model(img_left_up)
                mask_ru = model(img_right_up)
                mask_ld = model(img_left_down)
                mask_rd = model(img_right_down)
                # mask = model(img)
                # lu
                mask_prob_lu = torch.sigmoid(mask_lu).squeeze(0)
                mask_prob_lu = tf(mask_prob_lu.cuda(device))
                inf_lu = mask_prob_lu.squeeze().cuda(device)
                # ru
                mask_prob_ru = torch.sigmoid(mask_ru).squeeze(0)
                mask_prob_ru = tf(mask_prob_ru.cuda(device))
                inf_ru = mask_prob_ru.squeeze().cuda(device)
                # ld
                mask_prob_ld = torch.sigmoid(mask_ld).squeeze(0)
                mask_prob_ld = tf(mask_prob_ld.cuda(device))
                inf_ld = mask_prob_ld.squeeze().cuda(device)
                # rd
                mask_prob_rd = torch.sigmoid(mask_rd).squeeze(0)
                mask_prob_rd = tf(mask_prob_rd.cuda(device))
                inf_rd = mask_prob_rd.squeeze().cuda(device)
                # mask_prob = torch.sigmoid(mask).squeeze(0)
                # mask_prob = tf(mask_prob.cuda(device))
                # inf = mask_prob.squeeze().cuda(device)
            else:
                mask_lu = model(img_left_up)
                mask_lu_2 = model_2(img_left_up)
                mask_ru = model(img_right_up)
                mask_ru_2 = model_2(img_right_up)
                mask_ld = model(img_left_down)
                mask_ld_2 = model_2(img_left_down)
                mask_rd = model(img_right_down)
                mask_rd_2 = model_2(img_right_down)
                # mask = model(img)
                # mask_2 = model_2(img)

                # lu
                mask_prob_lu = torch.sigmoid(mask_lu).squeeze(0)
                mask_prob_lu = tf(mask_prob_lu.cuda(device))
                mask_prob_lu_2 = torch.sigmoid(mask_lu_2).squeeze(0)
                mask_prob_lu_2 = tf(mask_prob_lu_2.cuda(device))
                mask_prob_lu = (mask_prob_lu + mask_prob_lu_2) * 0.5
                inf_lu = mask_prob_lu.squeeze().cuda(device)

                # ru
                mask_prob_ru = torch.sigmoid(mask_ru).squeeze(0)
                mask_prob_ru = tf(mask_prob_ru.cuda(device))
                mask_prob_ru_2 = torch.sigmoid(mask_ru_2).squeeze(0)
                mask_prob_ru_2 = tf(mask_prob_ru_2.cuda(device))
                mask_prob_ru = (mask_prob_ru + mask_prob_ru_2) * 0.5
                inf_ru = mask_prob_ru.squeeze().cuda(device)

                # ld
                mask_prob_ld = torch.sigmoid(mask_ld).squeeze(0)
                mask_prob_ld = tf(mask_prob_ld.cuda(device))
                mask_prob_ld_2 = torch.sigmoid(mask_ld_2).squeeze(0)
                mask_prob_ld_2 = tf(mask_prob_ld_2.cuda(device))
                mask_prob_ld = (mask_prob_ld + mask_prob_ld_2) * 0.5
                inf_ld = mask_prob_ld.squeeze().cuda(device)

                # rd
                mask_prob_rd = torch.sigmoid(mask_rd).squeeze(0)
                mask_prob_rd = tf(mask_prob_rd.cuda(device))
                mask_prob_rd_2 = torch.sigmoid(mask_rd_2).squeeze(0)
                mask_prob_rd_2 = tf(mask_prob_rd_2.cuda(device))
                mask_prob_rd = (mask_prob_rd + mask_prob_rd_2) * 0.5
                inf_rd = mask_prob_rd.squeeze().cuda(device)

                # mask_prob = torch.sigmoid(mask).squeeze(0)
                # mask_prob = tf(mask_prob.cuda(device))
                # mask_prob_2 = torch.sigmoid(mask_2).squeeze(0)
                # mask_prob_2 = tf(mask_prob_2.cuda(device))
                # mask_prob = ( mask_prob + mask_prob_2 ) / 2
                # inf = mask_prob.squeeze().cuda(device)

        ## inf concat
        inf_left = torch.cat((inf_lu[0:260, ], inf_ld[12:, ]), dim=0)
        inf_right = torch.cat((inf_ru[0:260, ], inf_rd[12:, ]), dim=0)
        inf = torch.cat((inf_left, inf_right), dim=1)
        tmp_IoU = iou_pytorch(inf, gt, device)
        tmp_precision, tmp_recall = precision_recall_pytorch(inf, gt, device)

        IoU_list.append(tmp_IoU.to('cpu').item())
        precision_list.append(tmp_precision.to('cpu').item())
        recall_list.append(tmp_recall.to('cpu').item())

    if logfilePath is not None:
        with open(logfilePath, mode='a') as f:
            f.write('img num, IoU\n')
            for i, imgIoU in enumerate(IoU_list):
                f.write(f'{i:0>3}, {imgIoU}\n')
            f.write('\n')

    return IoU_list, precision_list, recall_list


def get_args():
    parser = argparse.ArgumentParser(description='Inference the UNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset-dir', metavar='D', type=str, default='./LIVECell_dataset',
                        help='Dataset directory path', dest='dataset_dir')
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
                        help='First num of kernels', dest='first_num_of_kernels')
    parser.add_argument('-c', '--checkpoint', metavar='C', type=str, nargs='?', default=None,
                        help='the path of segmenter', dest='checkpoint')
    parser.add_argument('-o', '--output-dir', metavar='O', type=str, nargs='?', default='trial',
                        help='output directory?', dest='output_dir')
    parser.add_argument('-cell', '--inference-cell', metavar='CN', type=str, nargs='?', default='mcf7',
                        help='inference cell name', dest='inference_cell')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu number?', dest='gpu_no')
    parser.add_argument('-scaling', '--scaling-type', type=str, nargs='?', default='unet',
                        help='scaling type?', dest='scaling_type')
    parser.add_argument('-coteaching', type=int, nargs='?', default=0,
                        help='inference co_teaching model?', dest='coteaching')
    parser.add_argument('-ensemble', type=int, nargs='?', default=0,
                        help='inference co_teaching model?', dest='ensem')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_no}'
    device = torch.device(f'cuda:{args.gpu_no}' if torch.cuda.is_available() else 'cpu')

    checkPoint = torch.load(args.checkpoint, map_location=device)

    # load U-Net
    net = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net.load_state_dict(checkPoint['best_net1'])
    net.to(device=device)
    net.eval()
    net_2 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_2.load_state_dict(checkPoint['best_net2'])
    net_2.to(device=device)
    net_2.eval()

    # create the result file
    dir_datasets = f'{args.dataset_dir}'
    dir_result = f'{args.output_dir}'
    dir_imgs = f'{dir_result}/imgs'
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_imgs, exist_ok=True)
    path_w = f'{dir_result}/evaluation.txt'

    # load test paths
    with open(path_w, mode='w') as f:
        f.write(f'inference single cell({args.inference_cell}, \n model:{args.checkpoint})\n')

    testDir = f'{dir_datasets}/test_data/{args.inference_cell}'
    testFiles = sorted(glob.glob(f'{testDir}/*'), key=natural_keys)
    tests = create_testlist(testFiles, scaling_type=args.scaling_type)

    # inference specific cell type
    IoU, precision, recall = eval_mcd(device, tests, model=net, model_2=net_2, logfilePath=path_w)
    with open(path_w, mode='a') as f:
        f.write('----inference----\n')
        f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
        f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision),
                                                           statistics.stdev(precision)))
        f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))

    # # net1
    # IoU, precision, recall = eval_mcd(device, tests, model=net, logfilePath=path_w)
    # with open(path_w, mode='a') as f:
    #     f.write('----inference net_1----\n')
    #     f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
    #     f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision),
    #                                                        statistics.stdev(precision)))
    #     f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))
    #
    # # net2
    # IoU, precision, recall = eval_mcd(device, tests, model=net_2, logfilePath=path_w)
    # with open(path_w, mode='a') as f:
    #     f.write('----inference net_2----\n')
    #     f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
    #     f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision),
    #                                                        statistics.stdev(precision)))
    #     f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))