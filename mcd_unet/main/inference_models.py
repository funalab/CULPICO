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
from tqdm import tqdm
import time

### inference using U-Net model ###
def inference_unet( test_list, filenames, model, logfilePath=None ):
    IoU_list = []
    Dice_list = []

    tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    
    for i, image in enumerate( tqdm( test_list ) ):

        img = torch.from_numpy(image[0]).unsqueeze(0).cpu()    

        with torch.no_grad():

            mask = model(img)
            mask_prob = torch.sigmoid(mask).squeeze(0)

            mask_prob = tf(mask_prob.cpu())
            inf = mask_prob.squeeze().cpu().numpy()

        tmp_Dice, tmp_IoU = calc_IoU(inf, image[1][0])
        Dice_list.append(tmp_Dice)
        IoU_list.append(tmp_IoU)

    if logfilePath != None:
        with open(logfilePath, mode='a') as f:
            f.write('img num, IoU\n')
            for i, imgIoU in enumerate(IoU_list):
                f.write(f'{filenames[i]}, {imgIoU}\n')
            f.write('\n')

    return Dice_list, IoU_list

### inference using MCD-U-Net model ###
def inference_mcd_unet( test_list, filenames , net_g=None, net_s1=None, net_s2=None, logfilePath=None ):
    IoU_list = []
    Dice_list = []
    tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    for i, image in enumerate( tqdm( test_list ) ):

        img = torch.from_numpy(image[0]).unsqueeze(0).cpu()

        with torch.no_grad():
            feat = net_g(img)
            mask_s1 = net_s1(*feat)
            mask_s2 = net_s2(*feat)

            mask_prob_s1 = torch.sigmoid(mask_s1).squeeze(0)
            mask_prob_s1 = tf(mask_prob_s1.cpu())
            mask_prob_s2 = torch.sigmoid(mask_s2).squeeze(0)
            mask_prob_s2 = tf(mask_prob_s2.cpu())

            inf_s1 = mask_prob_s1.squeeze().cpu().numpy()
            inf_s2 = mask_prob_s2.squeeze().cpu().numpy()
            inf = ( inf_s1 + inf_s2 ) / 2

        tmp_Dice, tmp_IoU = calc_IoU(inf, image[1][0])
        Dice_list.append(tmp_Dice)
        IoU_list.append(tmp_IoU)

    if logfilePath != None:
        with open(logfilePath, mode='a') as f:
            f.write('img num, IoU\n')
            for i, imgIoU in enumerate(IoU_list):
                # Write the file name and corresponding IoU
                f.write(f'{filenames[i]}, {imgIoU}\n')
            f.write('\n')

    return Dice_list, IoU_list

### calculate IoU & Dice ###
def calc_IoU(inf, mask):
    for i in range(inf.shape[0]):
        for j in range(inf.shape[1]):
            if inf[i][j]>0.5:
                inf[i][j] = 1
            else:
                inf[i][j] = 0
    inf = np.uint8(inf)

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

### merge inference image & ground truth ###
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

def writingResults( cellname, Dice, IoU, filePath ):
        with open( filePath, mode='a' ) as f:
            f.write(f'{cellname} inference result...\n')    
            f.write('Dice : {: .04f} +-{: .04f}\n'.format(statistics.mean(Dice), statistics.stdev(Dice)))
            f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))

def get_args():
    parser = argparse.ArgumentParser(description='Inference the UNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
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
    parser.add_argument('-cell1', '--cell-1', type=str, nargs='?', default='bt474',
                        help='inference cell name (1) ?', dest='cell_1')
    parser.add_argument('-cell2', '--cell-2', type=str, nargs='?', default=None,
                        help='inference cell name (2) ?', dest='cell_2')
    parser.add_argument('-raw', '--raw-mode', type=bool, nargs='?', default=0,
                        help='inference unet or mcd-unet?', dest='raw_mode')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.raw_mode == False:
        checkPoint = torch.load( args.checkpoint, map_location='cpu' )
        net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_g.load_state_dict( checkPoint['best_g'] )
        net_g.eval()
        net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_s1.load_state_dict( checkPoint['best_s1'] )
        net_s1.eval()
        net_s2 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_s2.load_state_dict( checkPoint['best_s2'] )
        net_s2.eval()
    else:
        # loading unet model
        model = UNet( first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True )
        model.load_state_dict( torch.load( args.checkpoint, map_location='cpu' ) )
        model.eval()

    # load first inference cell images
    testDir_1 = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data/{args.cell_1}'
    testFiles_1 = glob.glob(f'{testDir_1}/*')
    tests_1 = create_trainlist( testFiles_1, test=1, cut=1 )

    #cell_set_1 = []
    #imgsDir='/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data/shsy5y/test_set_128'
    #filepathList = glob.glob(f'{imgsDir}/*')

    if args.cell_2 != None:
        # load second inference cell images
        testDir_2 = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data/{args.cell_2}'
        testFiles_2 = glob.glob(f'{testDir_2}/*')
        tests_2 = create_trainlist( testFiles_2, test=1, cut=1 )

    # create the result file
    dir_result = './infResult/eval_{}_fk{}'.format( args.out_dir, args.first_num_of_kernels )
    #dir_imgs = f'{dir_result}/imgs'
    os.makedirs( dir_result, exist_ok=True )
    #os.makedirs( dir_imgs, exist_ok=True )
    path_w = f'{dir_result}/evaluation.txt'

    print( f'model: {args.checkpoint}' )

    if args.raw_mode == False:
        print( f'start infernce {args.cell_1}' )
        Dice, IoU = inference_mcd_unet( tests_1, testFiles_1 , net_g=net_g, net_s1=net_s1, net_s2=net_s2, logfilePath=path_w )
        writingResults( args.cell_1, Dice, IoU, path_w )
        if args.cell_2 != None:
            print( f'start infernce {args.cell_2}' )
            Dice, IoU = inference_mcd_unet( tests_2, testFiles_2 , net_g=net_g, net_s1=net_s1, net_s2=net_s2, logfilePath=path_w )
            writingResults( args.cell_2, Dice, IoU, path_w ) 
    else:
        print( f'start infernce {args.cell_1}' )
        Dice, IoU = inference_unet( tests_1, testFiles_1, model=model, logfilePath=path_w )
        writingResults( args.cell_2, Dice, IoU, path_w )
        if args.cell_2 != None:
            print( f'start infernce {args.cell_2}' )
            Dice, IoU = inference_unet( tests_2, testFiles_2, model=model, logfilePath=path_w )
            writingResults( args.cell_2, Dice, IoU, path_w ) 

    #img_result, img_merge = segment(seg_shsy5y, net_g=net_g, net_s=net_s1, use_mcd=1)
    

    #io.imsave(f'{dir_imgs}/input.tif', imgSet[-2].reshape(img.shape[-2], img.shape[-1]))
    #io.imsave(f'{dir_imgs}/result.tif', img_result)
    #io.imsave(f'{dir_imgs}/merge.tif', img_merge)    









