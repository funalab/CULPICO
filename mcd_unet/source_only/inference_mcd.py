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

def eval_unet(test_list, model=None, net_g=None, net_s=None, use_mcd=False, logfilePath=None):
    IoU_list = []
    Dice_list = []
    #現状1枚inference用コード->複数枚mean,std算出に対応させる必要
    #対応させたよ
    for i, image in enumerate(test_list):
    #print(i)
    #print(file[1].shape)
    #.tocuda()
        img = torch.from_numpy(image[0]).unsqueeze(0).cpu()
    
        with torch.no_grad():
            if use_mcd:
                feat = net_g(img)
                mask = net_s(*feat)
            else:
                mask = model(img)
                
            mask_prob = torch.sigmoid(mask).squeeze(0)
            
            tf = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize(h),
                transforms.ToTensor()
            ])
        
            mask_prob = tf(mask_prob.cpu())

            #mask_prob_np
            inf = mask_prob.squeeze().cpu().numpy()

        tmp_Dice, tmp_IoU = calc_IoU(inf, image[1][0])
        Dice_list.append(tmp_Dice)
        IoU_list.append(tmp_IoU)

    if logfilePath != None:
        with open(logfilePath, mode='w') as f:
            f.write('img num, IoU\n')
            for i, imgIoU in enumerate(IoU_list):
                f.write(f'{i:0>3}, {imgIoU}\n')
            f.write('\n')

    return Dice_list, IoU_list

def segment(test_list, model=None, net_g=None, net_s=None, use_mcd=False):
    
    for i, image in enumerate(test_list):
    #print(i)
    #print(file[1].shape)
    #.tocuda()
        img = torch.from_numpy(image[0]).unsqueeze(0).cpu()
    
        with torch.no_grad():
            if use_mcd:
                feat = net_g(img)
                mask = net_s(*feat)
            else:
                mask = model(img)
                
            mask_prob = torch.sigmoid(mask).squeeze(0)
            
            tf = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize(h),
                transforms.ToTensor()
            ])
        
            mask_prob = tf(mask_prob.cpu())

            #mask_prob_np
            inf = mask_prob.squeeze().cpu().numpy()

            for i in range(inf.shape[0]):
                for j in range(inf.shape[1]):
                    if inf[i][j]>0.5:
                        inf[i][j] = 1
                    else:
                        inf[i][j] = 0
            inf = np.uint8(inf)
            re, me = merge_images(image[1][0], inf)
            return re, me


def calc_IoU(inf, mask):
    for i in range(inf.shape[0]):
        for j in range(inf.shape[1]):
            if inf[i][j]>0.5:
                inf[i][j] = 1
            else:
                inf[i][j] = 0
    inf = np.uint8(inf)

    #inf.shape = gt.shape = (128, 128)
    #Type = np.ndarray
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    #temporary
    judge = inf - np.uint8(mask)
    #test_list[0][1][0]
    for i in range(inf.shape[0]):
        for j in range(inf.shape[1]):
            if judge[i][j] < 0:
                FN += 1
            elif judge[i][j] > 0:
                FP += 1
            else:
                if inf[i][j] > 0:
                    TP += 1
                else:
                    TN += 1

    Dice = 2 * TP / (2 * TP + FP + FN)
    IoU = TP / (TP + FP + FN)

    return Dice, IoU

def merge_images(img_gt: np.ndarray, img_segmented: np.ndarray):
    #img_segmented = make_size_equal(img_segmented, img_gt)

    img_segmented: np.ndarray = img_segmented > 0
    img_gt: np.ndarray = img_gt > 0

    result_img = np.zeros((img_gt.shape[0], img_gt.shape[1], 3))

    fp_img = np.logical_and(img_segmented, ~img_gt) * 255
    tp_img = np.logical_and(img_segmented, img_gt) * 255
    fn_img = np.logical_and(~img_segmented, img_gt) * 255

    result_img[:, :, 0] += fp_img  # R
    result_img[:, :, 1] += tp_img  # G
    result_img[:, :, 1] += fn_img  # G
    result_img[:, :, 2] += fn_img  # B

    return result_img.astype(np.uint8), img_segmented.astype(np.uint8) * 255

def get_args():
    parser = argparse.ArgumentParser(description='Inference the UNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=32,
                        help='First num of kernels', dest='first_num_of_kernels')
    #parser.add_argument('-m', '--model', metavar='M', type=str, nargs='?',
                        #help='the path of model', dest='path_of_model')
    parser.add_argument('-g', '--generator', metavar='G', type=str, nargs='?', default=None,
                        help='the path of generator', dest='path_of_g')
    parser.add_argument('-s', '--segmenter', metavar='S1', type=str, nargs='?', default=None,
                        help='the path of segmenter', dest='path_of_s')
    parser.add_argument('-c', '--checkpoint', metavar='C', type=str, nargs='?', default=None,
                        help='the path of segmenter', dest='checkpoint')
    parser.add_argument('-o', '--output', metavar='O', type=str, nargs='?', default='infResult',
                        help='out_dir?', dest='out_dir')
    parser.add_argument('-cell', '--cell', metavar='CN', type=str, nargs='?', default='shsy5y',
                        help='cell name', dest='cell')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    checkPoint = torch.load( args.checkpoint, map_location='cpu' )
    net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_g.load_state_dict( checkPoint['best_g'] )
    net_g.eval()
    net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
    net_s1.load_state_dict( checkPoint['best_s1'] )
    net_s1.eval()
    #net_s2.load_state_dict( checkPoint['best_s2'] )
    #net_s2.eval()

    #load test images
    # test images (520, 704) -> (272, 352)  ##(260, 352) -> (272, 352)##
    testDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data/{args.cell}'
    testFiles = glob.glob(f'{testDir}/*')
    
    tests = create_trainlist( testFiles, test=1, cut=1 )
    
    #load image for segmentation
    seg_HeLa = []
    seg_NIH = []

    #create the result file
    dir_result = './infResult/eval_{}_fk{}'.format( args.out_dir, args.first_num_of_kernels )
    dir_imgs = f'{dir_result}/imgs'
    os.makedirs( dir_result, exist_ok=True )
    os.makedirs( dir_imgs, exist_ok=True )
    path_w = f'{dir_result}/evaluation.txt'
    
    Dice, IoU = eval_unet(tests, net_g=net_g, net_s=net_s1, use_mcd=1, logfilePath=path_w)
    
    #img_result, img_merge = segment(seg_HeLa, net_g=net_g, net_s=net_s1, use_mcd=1)
    
    with open(path_w, mode='a') as f:
        f.write('Dice : {: .04f} +-{: .04f}\n'.format(statistics.mean(Dice), statistics.stdev(Dice)))
        f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))

    #io.imsave(f'{dir_imgs}/result.tif', img_result)
    #io.imsave(f'{dir_imgs}/merge.tif', img_merge)    
