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

def eval_mcd( device, test_list, model=None, net_g=None, net_s=None, net_s_another=None, raw=False, logfilePath=None):

    IoU_list = []

    tf = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize(h),
                transforms.ToTensor()
            ])

    for i, image in enumerate(test_list):

        img = torch.from_numpy(image[0]).unsqueeze(0).cuda(device)
        gt = torch.from_numpy(image[1][0]).cuda(device)
    
        with torch.no_grad():
            if raw:
                mask = model(img)

                mask_prob = torch.sigmoid(mask).squeeze(0)
                mask_prob = tf(mask_prob.cuda(device))
                inf = mask_prob.squeeze().cuda(device)

            else:
                feat = net_g(img)
                mask = net_s(*feat)
                mask_ano = net_s_another(*feat)

                mask_prob = torch.sigmoid(mask).squeeze(0)
                mask_prob_ano = torch.sigmoid(mask_ano).squeeze(0)

                mask_prob = tf(mask_prob.cuda(device))
                mask_prob_ano = tf( mask_prob_ano.cuda(device) )

                inf_s1 = mask_prob.squeeze().cuda(device)
                inf_s2 = mask_prob_ano.squeeze().cuda(device)

                inf = (inf_s1 + inf_s2) /2


        tmp_IoU = iou_pytorch(inf, gt, device)

        IoU_list.append(tmp_IoU.to('cpu').item())

    if logfilePath != None:
        with open(logfilePath, mode='w') as f:
            f.write('img num, IoU\n')
            for i, imgIoU in enumerate(IoU_list):
                f.write(f'{i:0>3}, {imgIoU}\n')
            f.write('\n')

    return IoU_list


def get_args():
    parser = argparse.ArgumentParser(description='Inference the UNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-fk', '--first-kernels', metavar='FK', type=int, nargs='?', default=64,
                        help='First num of kernels', dest='first_num_of_kernels')
    parser.add_argument('-c', '--checkpoint', metavar='C', type=str, nargs='?', default=None,
                        help='the path of segmenter', dest='checkpoint')
    parser.add_argument('-o', '--output', metavar='O', type=str, nargs='?', default='infResult',
                        help='out_dir?', dest='out_dir')
    parser.add_argument('-cell', '--cell', metavar='CN', type=str, nargs='?', default='shsy5y',
                        help='cell name', dest='cell')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu_num?', dest='gpu_num')
    parser.add_argument('-raw', '--raw-unet', type=bool, nargs='?', default=0,
                        help='train raw unet?', dest='raw_mode')
    parser.add_argument('-scaling', '--scaling-type', type=str, nargs='?', default='unet',
                        help='scaling type?', dest='scaling_type')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    checkPoint = torch.load( args.checkpoint, map_location=device )

    if args.raw_mode:
        # load U-Net
        net = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        #print( checkPoint.keys() )
        net.load_state_dict( checkPoint['best_net'] )
        #net.load_state_dict( checkPoint )
        net.to(device=device)
        net.eval()
        net_g=None; net_s1=None; net_s2=None

    else:
        # load MCD-U-Net
        net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_g.load_state_dict( checkPoint['best_g'] )
        net_g.to(device=device)
        net_g.eval()
        net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_s1.load_state_dict( checkPoint['best_s1'] )
        net_s1.to(device=device)
        net_s1.eval()
        net_s2 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
        net_s2.load_state_dict( checkPoint['best_s2'] )
        net_s2.to(device=device)
        net_s2.eval()
    
    testDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data/{args.cell}'
    testFiles = glob.glob(f'{testDir}/*')
    
    tests = create_trainlist( testFiles, scaling_type=args.scaling_type, test=1, cut=1 )
    
    seg_shsy5y = []
    imgsDir='/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data/shsy5y/test_set_128'
    filepathList = glob.glob(f'{imgsDir}/*')

    cut=1; test=1;
    imgSet = [0] * 2
    for filePath in filepathList:
        img = io.imread( filePath )
        if 'Phase' in filePath:

            if args.scaling_type == "unet":
                img = scaling_image(img)
                img = img - np.median(img)
            elif args.scaling_type == "standard":
                img = standardize_image(img)
            elif args.scaling_type == "normal":
                img = scaling_image(img)
            
            if cut: img = img[130:390, 176:528]
            imgSet[-2] = img if test == False else img.reshape([1, img.shape[-2], img.shape[-1]])
 
        else:
            img = img / 255
            if cut: img = img[130:390, 176:528]
            imgSet[-1] = img if test == False else img.reshape([1, img.shape[-2], img.shape[-1]])
    seg_shsy5y.append(imgSet)

    # create the result file
    dir_result = './infResult/eval_{}_fk{}'.format( args.out_dir, args.first_num_of_kernels )
    dir_imgs = f'{dir_result}/imgs'
    os.makedirs( dir_result, exist_ok=True )
    os.makedirs( dir_imgs, exist_ok=True )
    path_w = f'{dir_result}/evaluation.txt'
    #net_s_another=net_s2,
    IoU = eval_mcd( device, tests, model=net, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
    
    #img_result, img_merge = segment(seg_shsy5y, net_g=net_g, net_s=net_s1, use_mcd=1)
    
    with open(path_w, mode='a') as f:
        for i, filename in enumerate(testFiles):
            f.write('{}:{}\n'.format( i, filename ))
            
        #f.write('Dice : {: .04f} +-{: .04f}\n'.format(statistics.mean(Dice), statistics.stdev(Dice)))
        f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))

    #io.imsave(f'{dir_imgs}/input.tif', imgSet[-2].reshape(img.shape[-2], img.shape[-1]))
    #io.imsave(f'{dir_imgs}/result.tif', img_result)
    #io.imsave(f'{dir_imgs}/merge.tif', img_merge)
