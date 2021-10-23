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
import statistics


def eval(test_list, split_flag=False):
    IoU_list = []
    Dice_list = []
    if split_flag:
        adjust_size = 250
    else:
        adjust_size = 500
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
            
    
        inf = adjust_img( inf, adjust_size, adjust_size )
        tmp_Dice, tmp_IoU = calc_IoU(inf, image[1][0])
        Dice_list.append(tmp_Dice)
        IoU_list.append(tmp_IoU)

    
    return Dice_list, IoU_list

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
    parser.add_argument('-scaling', '--scaling_type', metavar='ST', type=str, nargs='?', default='unet',
                        help='scaling method?', dest='scaling_type')
    parser.add_argument('-mk', '--marker', metavar='FM', type=str, nargs='?', default=None,
                        help='?', dest='marker')
    parser.add_argument('-se', '--start-epoch', metavar='SE', type=int, nargs='?', default=275,
                        help='?', dest='start_epoch')
    parser.add_argument('-cpn', '--cp-name', metavar='CPN', type=str, nargs='?', default=None,
                        help='check point name before "_epoch" ', dest='cp_name')
    parser.add_argument('-sp', '--split', metavar='SP', type=bool, nargs='?', default=False,
                        help='split mode T or F?" ', dest='split_flag')



    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    tests = []
    name = "phase"
    #necessary to be absolute path
    
    if args.split_flag:
        test_files = glob.glob("../../dataset_smiyaki/test_split/HeLa/*")
        mp_size = 256
    else:
        test_files = glob.glob("../../dataset_smiyaki/test_raw/HeLa/*")
        mp_size = 512
    
    #test_files = glob.glob("../../dataset_smiyaki/test_raw/HeLa/*")
    for testfile in test_files:
        ph_lab = [0] * 2
        #*set*/
        path_phase_and_lab = glob.glob(f"{testfile}/*")
        #print(f"{trainfile}")
        for path_img in path_phase_and_lab:
            #print("hoge")
            img = io.imread(path_img)
            if name in path_img:
                """
                
                """
                #original unet scaling (subtracting by median)
                img = scaling_image(img)
                if args.scaling_type == 'unet':
                    img = img - np.median( img )

                img = mirror_padding( img, mp_size, mp_size )
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
                """
                if args.split_flag:
                    #未mirror_padding
                    #未reshape( (512, 512) -> (1, 512, 512) )
                    ph_lab[0] = img
                else:
                    mp_size = 512
                    img = mirror_padding( img, mp_size, mp_size )
                    ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
                """
                #img = img - np.median(img)
                
            else:
                img = img / 255
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
                """
                if args.split_flag:
                    #未reshape
                    ph_lab[1] = img
                else:
                    #img = mirror_padding( img, 512, 512 )
                
                    ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
                """
        tests.append(ph_lab)
    """ 
    if args.split_flag:
    #data整形
        mp_size = 256
        #等分割
        tests = cutting_img( tests[0], 250 )
        for split_imgs in tests:
            #phase(split_imgs[0]):mirror_padding & reshape
            tmp_phase = split_imgs[0]
            tmp_phase_mp = mirror_padding( tmp_phase, mp_size, mp_size )
            split_imgs[0] = tmp_phase_mp.reshape( [1, tmp_phase_mp.shape[-2], tmp_phase_mp.shape[-1]] ) 
            #label(split_imgs[1]):reshape
            tmp_lab = split_imgs[1]
            split_imgs[1] = tmp_lab.reshape( [1, tmp_lab.shape[-2], tmp_lab.shape[-1]] ) 
    """
    
    NIHtests = []
    name = "phase"
    #necessary to be absolute path
    
    if args.split_flag:
        test_files = glob.glob("../../dataset_smiyaki/test_split/3T3/*")
        mp_size = 256
    else:
        test_files = glob.glob("../../dataset_smiyaki/test_raw/3T3/*")
        mp_size = 512
    
    #test_files = glob.glob("../../dataset_smiyaki/test_raw/3T3/*")
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

                img = mirror_padding( img, mp_size, mp_size )
                ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
                """
                if args.split_flag:
                    #未mirror_padding
                    #未reshape( (512, 512) -> (1, 512, 512) )
                    ph_lab[0] = img
                else:
                    mp_size = 512
                    img = mirror_padding( img, mp_size, mp_size )
                    ph_lab[0] = img.reshape([1, img.shape[-2], img.shape[-1]])
                """
                
            else:
                img = img / 255
                #img = mirror_padding( img, 512, 512 )
                ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])
                """
                if args.split_flag:
                    #未reshape
                    ph_lab[1] = img
                else:
                    #img = mirror_padding( img, 512, 512 )
                    ph_lab[1] = img.reshape([1, img.shape[-2], img.shape[-1]])                
                """
        NIHtests.append(ph_lab)
    """
    if args.split_flag:
        
        mp_size = 256
        #等分割
        NIHtests = cutting_img( NIHtests[0], 250 )
        for split_imgs in NIHtests:
            #phase(split_imgs[0]):mirror_padding & reshape
            tmp_phase = split_imgs[0]
            tmp_phase = mirror_padding( tmp_phase, mp_size, mp_size )
            split_imgs[0] = tmp_phase.reshape( [1, tmp_phase.shape[-2], tmp_phase.shape[-1]] ) 
            #label(split_imgs[1]):reshape
            tmp_lab = split_imgs[1]
            split_imgs[1] = tmp_lab.reshape( [1, tmp_lab.shape[-2], tmp_lab.shape[-1]] ) 
    """

    #CP_HeLa_Adam_epoch500_fk64_b1.pth

    if args.model_dir != None:
        path_w = f'./eval_by_epoch_{args.marker}.txt'
        for i in range(args.start_epoch, 1500):
            path_model = f'{args.model_dir}/CP_HeLa_Adam_epoch{i+1}_fk{args.first_num_of_kernels}_b4.pth'
            #CP_HeLa_Adam_epoch300_fk64_b2.pth
            #CP_HeLa_Adam_epoch98_fk32_b1.pth
            #path_model = f'{args.model_dir}/CP_HeLa_Adam_epoch{i+1}_fk64_b2.pth'
            #CP_HeLa_SGD_epoch458_fk64_b2.pth
            model = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            model.load_state_dict(torch.load(path_model, map_location='cpu'))
            model.eval()

            HeLa_Dice, HeLa_IoU = eval(tests, split_flag=args.split_flag)
            NIH_Dice, NIH_IoU = eval(NIHtests, split_flag=args.split_flag)
            #print(HeLa_IoU)
            #print(NIH_IoU)
            with open(path_w, mode='a') as f:
                f.write('epoch : {}\n'.format(i+1))
                f.write('HeLaIoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(HeLa_IoU), statistics.stdev(HeLa_IoU)))
                f.write('NIHIoU : {: .04f} +-{: .04f}\n\n'.format(statistics.mean(NIH_IoU), statistics.stdev(NIH_IoU)))
                

    else:
        """
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
        """
    
