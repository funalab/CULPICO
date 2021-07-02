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

def eval_unet(test_list, model=None, net_g=None, net_s=None, use_mcd=False):
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
    parser.add_argument('-mcd', '--use_mcdmodel', metavar='MCD', type=bool, nargs='?', default=False,
                        help='use mcdmodel?', dest='mcd')
    parser.add_argument('-g', '--generator', metavar='G', type=str, nargs='?', default=None,
                        help='the path of generator', dest='path_of_g')
    parser.add_argument('-s', '--segmenter', metavar='S1', type=str, nargs='?', default=None,
                        help='the path of segmenter', dest='path_of_s')
    parser.add_argument('-c', '--checkpoint', metavar='C', type=str, nargs='?', default=None,
                        help='the path of segmenter', dest='checkpoint')
    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    #load test images
    tests = []
    name = "phase"
    #necessary to be absolute path
    test_files = glob.glob("dataset_smiyaki/testing_data/HeLa/*")
    for testfile in test_files:
        ph_lab = [0] * 2
        #*set*/
        path_phase_and_lab = glob.glob(f"{testfile}/*")
        #print(f"{trainfile}")
        for path_img in path_phase_and_lab:
            #print("hoge")
            img = io.imread(path_img)
            if name in path_img:
                #original unet scaling (subtracting by median)
                img = scaling_image(img)
                #img = img - np.median(img)
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
        tests.append(ph_lab)

    NIHtests = []
    #necessary to be absolute path
    test_files = glob.glob("dataset_smiyaki/testing_data/3T3/*")
    for testfile in test_files:
        ph_lab = [0] * 2
        #*set*/
        path_phase_and_lab = glob.glob(f"{testfile}/*")
        #print(f"{trainfile}")
        for path_img in path_phase_and_lab:
            #print("hoge")
            img = io.imread(path_img)
            if name in path_img:
                #original unet scaling (subtracting by median)
                img = scaling_image(img)
                #img = img - np.median(img)
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
        NIHtests.append(ph_lab)
        
    #load image for segmentation
    seg_HeLa = []
    #necessary to be absolute path
    test_files = glob.glob("dataset_smiyaki/test_data/HeLa/*")
    for testfile in test_files:
        ph_lab = [0] * 2
        #*set*/
        path_phase_and_lab = glob.glob(f"{testfile}/*")
        #print(f"{trainfile}")
        for path_img in path_phase_and_lab:
            #print("hoge")
            img = io.imread(path_img)
            if name in path_img:
                #original unet scaling (subtracting by median)
                img = scaling_image(img)
                #img = img - np.median(img)
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
        seg_HeLa.append(ph_lab)

    seg_NIH = []
    #necessary to be absolute path
    test_files = glob.glob("dataset_smiyaki/test_data/3T3/*")
    for testfile in test_files:
        ph_lab = [0] * 2
        #*set*/
        path_phase_and_lab = glob.glob(f"{testfile}/*")
        #print(f"{trainfile}")
        for path_img in path_phase_and_lab:
            #print("hoge")
            img = io.imread(path_img)
            if name in path_img:
                #original unet scaling (subtracting by median)
                img = scaling_image(img)
                #img = img - np.median(img)
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
        seg_NIH.append(ph_lab)

    #create the result file
    if args.mcd :
        dir_result = './eval_murata'

    else:
        dir_result = './eval_{}'.format(args.path_of_model)
    os.makedirs(dir_result, exist_ok=True)
    path_w = f'{dir_result}/evaluation.txt'
    
    if args.mcd:
        """
        #checkpoint = torch.load(args.path_of_model, map_location='cpu')
        net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        #net_g.load_state_dict(checkpoint['best_g'])
        net_g.load_state_dict(torch.load(args.path_of_g, map_location='cpu'))
        net_g.eval()
        
        net_s = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        #net_s.load_state_dict(checkpoint['best_s1'])
        net_s.load_state_dict(torch.load(args.path_of_s, map_location='cpu'))
        net_s.eval()
        HeLa_Dice, HeLa_IoU = eval_unet(tests, net_g=net_g, net_s=net_s, use_mcd=args.mcd)
        NIH_Dice, NIH_IoU = eval_unet(NIHtests, net_g=net_g, net_s=net_s, use_mcd=args.mcd)
        HeLa_result, HeLa_merge = segment(seg_HeLa, net_g=net_g, net_s=net_s, use_mcd=args.mcd)
        NIH_result, NIH_merge = segment(seg_NIH, net_g=net_g, net_s=net_s, use_mcd=args.mcd)
        """
        for i in range(200):
            path_g = f'{args.checkpoint}/CP_G_epoch{i+1}.pth'
            path_s = f'{args.checkpoint}/CP_S_epoch{i+1}.pth' 
            net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net_g.load_state_dict(torch.load(path_g, map_location='cpu'))
            net_g.eval()
            net_s = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net_s.load_state_dict(torch.load(path_s, map_location='cpu'))
            net_s.eval()
            HeLa_Dice, HeLa_IoU = eval_unet(tests, net_g=net_g, net_s=net_s, use_mcd=args.mcd)
            NIH_Dice, NIH_IoU = eval_unet(NIHtests, net_g=net_g, net_s=net_s, use_mcd=args.mcd)
            print('epoch:{}'.format(i+1))
            print('HeLaIoU : {: .04f} +-{: .04f}'.format(statistics.mean(HeLa_IoU), statistics.stdev(HeLa_IoU)))
            print('NIHIoU : {: .04f} +-{: .04f}'.format(statistics.mean(NIH_IoU), statistics.stdev(NIH_IoU)))
    else:
        
        model_path = glob.glob(f"{args.path_of_model}/checkpoint/*")
        
        total_HeLa_IoU = []
        total_NIH_IoU = []
        for path in model_path:
            model = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
            HeLa_Dice, HeLa_IoU = eval_unet(tests, model=model)
            NIH_Dice, NIH_IoU = eval_unet(NIHtests, model=model)
            total_HeLa_IoU.extend(HeLa_IoU)
            total_NIH_IoU.extend(NIH_IoU)
        HeLa_result, HeLa_merge = segment(seg_HeLa, model=model)
        NIH_result, NIH_merge = segment(seg_NIH, model=model)
        
        
    with open(path_w, mode='w') as f:
        #f.write('HeLaDice : {: .04f} +-{: .04f}\n'.format(statistics.mean(HeLa_Dice), statistics.stdev(HeLa_Dice)))
        f.write('HeLaIoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(HeLa_IoU), statistics.stdev(HeLa_IoU)))
        #f.write('NIHDice : {: .04f} +-{: .04f}\n'.format(statistics.mean(NIH_Dice), statistics.stdev(NIH_Dice)))
        f.write('NIHIoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(NIH_IoU), statistics.stdev(NIH_IoU)))

    io.imsave(f'{dir_result}/HeLa_result.tif', HeLa_result)
    io.imsave(f'{dir_result}/HeLa_merge.tif', HeLa_merge)
    io.imsave(f'{dir_result}/NIH_result.tif', NIH_result)
    io.imsave(f'{dir_result}/NIH_merge.tif', NIH_merge)
    
    
