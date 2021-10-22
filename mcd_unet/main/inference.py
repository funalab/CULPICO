import argparse
from functions_io import *
from model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import glob
from skimage import io
import os

def eval(test_list):
    #現状1枚inference用コード->複数枚mean,std算出に対応させる必要
    for i, image in enumerate(test_list):
    #print(i)
    #print(file[1].shape)
    #.tocuda()
        img = torch.from_numpy(image[0]).unsqueeze(0).cpu()
        img = img.float()
        
        with torch.no_grad():
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

    
    inf = adjust_img( inf, 500, 500 )
    #(640, 640)->(500,500)
    #inf = inf[70:570, 70:570]
    label = test_list[0][1][0]
    #label = label[70:570, 70:570]
    #Type = np.ndarray
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    #temporary
    judge = inf - np.uint8(label)
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

    result, merge = merge_images(label, inf)
    return Dice, IoU, result, merge

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
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
                        help='First num of kernels', dest='first_num_of_kernels')
    parser.add_argument('-m', '--model', metavar='M', type=str, nargs='?',
                        help='the path of model', dest='Path_of_model')
    parser.add_argument('-md', '--modeldir', metavar='MD', type=str, nargs='?', default=None,
                        help='the path of the dir of model_list', dest='model_dir')
    parser.add_argument('-scaling', '--scaling_type', metavar='ST', type=str, nargs='?', default=None,
                        help='scaling method?', dest='scaling_type')
    parser.add_argument('-mk', '--marker', metavar='FM', type=str, nargs='?', default=None,
                        help='?', dest='marker')
    parser.add_argument('-se', '--start-epoch', metavar='SE', type=int, nargs='?', default=275,
                        help='?', dest='start_epoch')
    parser.add_argument('-cpn', '--cp-name', metavar='CPN', type=str, nargs='?', default=None,
                        help='check point name before "_epoch" ', dest='cp_name')



    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    tests = []
    name = "phase"
    #necessary to be absolute path
    test_files = glob.glob("../../dataset_smiyaki/test_raw/HeLa/*")
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
                if args.scaling_type == 'unet':
                    img = img - np.median( img )
                img = mirror_padding( img, 512, 512 )
                #img = img - np.median(img)
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                #img = mirror_padding( img, 512, 512 )
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
        tests.append(ph_lab)

    NIHtests = []
    name = "phase"
    #necessary to be absolute path
    test_files = glob.glob("../../dataset_smiyaki/test_raw/3T3/*")
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
                if args.scaling_type == 'unet':
                    img = img - np.median( img )
                img = mirror_padding( img, 512, 512 )
                #img = img - np.median(img)
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                #img = mirror_padding( img, 512, 512 )
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
        NIHtests.append(ph_lab)

    print(len(NIHtests))

    #CP_HeLa_Adam_epoch500_fk64_b1.pth

    if args.model_dir != None:
        path_w = f'./eval_by_epoch_{args.marker}.txt'
        for i in range(args.start_epoch, 700):
            path_model = f'{args.model_dir}/{args.cp_name}_epoch{i+1}_fk64_b2.pth'
            #CP_HeLa_Adam_epoch300_fk64_b2.pth
            #path_model = f'{args.model_dir}/CP_HeLa_Adam_epoch{i+1}_fk64_b2.pth'
            #CP_HeLa_SGD_epoch458_fk64_b2.pth
            model = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            model.load_state_dict(torch.load(path_model, map_location='cpu'))
            model.eval()
            
            NIHDice, NIHIoU, result_NIH, merge_NIH = eval(NIHtests)
            Dice, IoU, result_HeLa, merge_HeLa = eval(tests)
            with open(path_w, mode='a') as f:
                f.write('epoch : {: .04f}\n'.format(i+1))
                f.write('HeLaIoU : {: .04f}\n'.format(IoU))
                f.write('NIHIoU : {: .04f}\n\n'.format(NIHIoU))

    else:
        model = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        model.load_state_dict(torch.load(args.Path_of_model, map_location='cpu'))
        model.eval()

        dir_result = './eval512_result'
        os.makedirs(dir_result, exist_ok=True)
        path_w = f'{dir_result}/evaluation.txt'
        
        Dice, IoU, result_HeLa, merge_HeLa = eval(tests)
        NIHDice, NIHIoU, result_NIH, merge_NIH = eval(NIHtests)
        with open(path_w, mode='w') as f:
            f.write('HeLaDice : {: .04f}\n'.format(Dice))
            f.write('HeLaIoU : {: .04f}\n'.format(IoU))
            f.write('NIHDice : {: .04f}\n'.format(NIHDice))
            f.write('NIHIoU : {: .04f}\n'.format(NIHIoU))

        io.imsave(f'{dir_result}/HeLa_result.tif', result_HeLa)
        io.imsave(f'{dir_result}/HeLa_merge.tif', merge_HeLa)
        io.imsave(f'{dir_result}/NIH_result.tif', result_NIH)
        io.imsave(f'{dir_result}/NIH_merge.tif', merge_NIH)
    
    
