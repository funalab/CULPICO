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

def eval_mcd( device, test_list, model=None, model_2=None, net_g=None, net_s=None, net_s_another=None, raw=False, logfilePath=None):

    IoU_list = []
    precision_list = []
    recall_list = []

    tf = transforms.Compose([
                transforms.ToPILImage(),
                #transforms.Resize(h),
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
        gt_left = torch.cat( (gt_left_up[0:260,], gt_left_down[12:,]), dim=0 )
        gt_right = torch.cat( (gt_right_up[0:260,], gt_right_down[12:,]), dim=0 )
        gt = torch.cat( (gt_left, gt_right), dim=1 )

    
        with torch.no_grad():
            if raw:
                if model_2 == None:
                    mask_lu = model(img_left_up)
                    mask_ru = model(img_right_up)
                    mask_ld = model(img_left_down)
                    mask_rd = model(img_right_down)
                    #mask = model(img)
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
                    #mask_prob = torch.sigmoid(mask).squeeze(0)
                    #mask_prob = tf(mask_prob.cuda(device))
                    #inf = mask_prob.squeeze().cuda(device)
                else:
                    mask_lu = model(img_left_up)
                    mask_lu_2 = model_2(img_left_up)
                    mask_ru = model(img_right_up)
                    mask_ru_2 = model_2(img_right_up)
                    mask_ld = model(img_left_down)
                    mask_ld_2 = model_2(img_left_down)
                    mask_rd = model(img_right_down)
                    mask_rd_2 = model_2(img_right_down)
                    #mask = model(img)
                    #mask_2 = model_2(img)

                    # lu
                    mask_prob_lu = torch.sigmoid(mask_lu).squeeze(0)
                    mask_prob_lu = tf(mask_prob_lu.cuda(device))
                    mask_prob_lu_2 = torch.sigmoid(mask_lu_2).squeeze(0)
                    mask_prob_lu_2 = tf(mask_prob_lu_2.cuda(device))
                    mask_prob_lu = ( mask_prob_lu + mask_prob_lu_2 ) * 0.5                   
                    inf_lu = mask_prob_lu.squeeze().cuda(device)

                    # ru
                    mask_prob_ru = torch.sigmoid(mask_ru).squeeze(0)
                    mask_prob_ru = tf(mask_prob_ru.cuda(device))
                    mask_prob_ru_2 = torch.sigmoid(mask_ru_2).squeeze(0)
                    mask_prob_ru_2 = tf(mask_prob_ru_2.cuda(device))
                    mask_prob_ru = ( mask_prob_ru + mask_prob_ru_2 ) * 0.5                   
                    inf_ru = mask_prob_ru.squeeze().cuda(device)

                    # ld
                    mask_prob_ld = torch.sigmoid(mask_ld).squeeze(0)
                    mask_prob_ld = tf(mask_prob_ld.cuda(device))
                    mask_prob_ld_2 = torch.sigmoid(mask_ld_2).squeeze(0)
                    mask_prob_ld_2 = tf(mask_prob_ld_2.cuda(device))
                    mask_prob_ld = ( mask_prob_ld + mask_prob_ld_2 ) * 0.5                   
                    inf_ld = mask_prob_ld.squeeze().cuda(device)

                    # rd
                    mask_prob_rd = torch.sigmoid(mask_rd).squeeze(0)
                    mask_prob_rd = tf(mask_prob_rd.cuda(device))
                    mask_prob_rd_2 = torch.sigmoid(mask_rd_2).squeeze(0)
                    mask_prob_rd_2 = tf(mask_prob_rd_2.cuda(device))
                    mask_prob_rd = ( mask_prob_rd + mask_prob_rd_2 ) * 0.5                   
                    inf_rd = mask_prob_rd.squeeze().cuda(device)

                    #mask_prob = torch.sigmoid(mask).squeeze(0)
                    #mask_prob = tf(mask_prob.cuda(device))
                    #mask_prob_2 = torch.sigmoid(mask_2).squeeze(0)
                    #mask_prob_2 = tf(mask_prob_2.cuda(device))
                    #mask_prob = ( mask_prob + mask_prob_2 ) / 2                    
                    #inf = mask_prob.squeeze().cuda(device)

            else:
                # lu
                feat_lu = net_g(img_left_up)
                mask_lu = net_s(*feat_lu)
                mask_prob_lu = torch.sigmoid(mask_lu).squeeze(0)
                mask_prob_lu = tf(mask_prob_lu.cuda(device))
                # ru
                feat_ru = net_g(img_right_up)
                mask_ru = net_s(*feat_ru)
                mask_prob_ru = torch.sigmoid(mask_ru).squeeze(0)
                mask_prob_ru = tf(mask_prob_ru.cuda(device))
                # ld
                feat_ld = net_g(img_left_down)
                mask_ld = net_s(*feat_ld)
                mask_prob_ld = torch.sigmoid(mask_ld).squeeze(0)
                mask_prob_ld = tf(mask_prob_ld.cuda(device))
                #rd
                feat_rd = net_g(img_right_down)
                mask_rd = net_s(*feat_rd)
                mask_prob_rd = torch.sigmoid(mask_rd).squeeze(0)
                mask_prob_rd = tf(mask_prob_rd.cuda(device))

                #feat = net_g(img)
                #mask = net_s(*feat)
                #mask_prob = torch.sigmoid(mask).squeeze(0)
                #mask_prob = tf(mask_prob.cuda(device))
                
                if net_s_another == None:
                    inf_lu = mask_prob_lu.squeeze().cuda(device)
                    inf_ru = mask_prob_ru.squeeze().cuda(device)
                    inf_ld = mask_prob_ld.squeeze().cuda(device)
                    inf_rd = mask_prob_rd.squeeze().cuda(device)
                else:
                    # lu
                    mask_ano_lu = net_s_another(*feat_lu)
                    mask_prob_ano_lu = torch.sigmoid(mask_ano_lu).squeeze(0)
                    mask_prob_ano_lu = tf( mask_prob_ano_lu.cuda(device) )
                    inf_s1_lu = mask_prob_lu.squeeze().cuda(device)
                    inf_s2_lu = mask_prob_ano_lu.squeeze().cuda(device)
                    inf_lu = (inf_s1_lu + inf_s2_lu) * 0.5
                    
                    # ru
                    mask_ano_ru = net_s_another(*feat_ru)
                    mask_prob_ano_ru = torch.sigmoid(mask_ano_ru).squeeze(0)
                    mask_prob_ano_ru = tf( mask_prob_ano_ru.cuda(device) )
                    inf_s1_ru = mask_prob_ru.squeeze().cuda(device)
                    inf_s2_ru = mask_prob_ano_ru.squeeze().cuda(device)
                    inf_ru = (inf_s1_ru + inf_s2_ru) * 0.5
                    
                    # ld
                    mask_ano_ld = net_s_another(*feat_ld)
                    mask_prob_ano_ld = torch.sigmoid(mask_ano_ld).squeeze(0)
                    mask_prob_ano_ld = tf( mask_prob_ano_ld.cuda(device) )
                    inf_s1_ld = mask_prob_ld.squeeze().cuda(device)
                    inf_s2_ld = mask_prob_ano_ld.squeeze().cuda(device)
                    inf_ld = (inf_s1_ld + inf_s2_ld) * 0.5

                    # rd
                    mask_ano_rd = net_s_another(*feat_rd)
                    mask_prob_ano_rd = torch.sigmoid(mask_ano_rd).squeeze(0)
                    mask_prob_ano_rd = tf( mask_prob_ano_rd.cuda(device) )
                    inf_s1_rd = mask_prob_rd.squeeze().cuda(device)
                    inf_s2_rd = mask_prob_ano_rd.squeeze().cuda(device)
                    inf_rd = (inf_s1_rd + inf_s2_rd) * 0.5
                    

                    #mask_ano = net_s_another(*feat)
                    #mask_prob_ano = torch.sigmoid(mask_ano).squeeze(0)
                    #mask_prob_ano = tf( mask_prob_ano.cuda(device) )
                    #inf_s1 = mask_prob.squeeze().cuda(device)
                    #inf_s2 = mask_prob_ano.squeeze().cuda(device)
                    #inf = (inf_s1 + inf_s2) /2
        
        ## inf concat
        inf_left = torch.cat( (inf_lu[0:260,], inf_ld[12:,]), dim=0 )
        inf_right = torch.cat( (inf_ru[0:260,], inf_rd[12:,]), dim=0 )
        inf = torch.cat( (inf_left, inf_right), dim=1 )
        tmp_IoU = iou_pytorch(inf, gt, device)
        tmp_precision, tmp_recall = precision_recall_pytorch(inf, gt, device)

        IoU_list.append(tmp_IoU.to('cpu').item())
        precision_list.append(tmp_precision.to('cpu').item())
        recall_list.append(tmp_recall.to('cpu').item())

    #各画像のIoUを出力
    """
    if logfilePath != None:
        with open(logfilePath, mode='a') as f:
            f.write('img num, IoU\n')
            for i, imgIoU in enumerate(IoU_list):
                f.write(f'{i:0>3}, {imgIoU}\n')
            f.write('\n')
    """
    return IoU_list, precision_list, recall_list


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
    parser.add_argument('-raw', '--raw-unet', type=int, nargs='?', default=0,
                        help='train raw unet?', dest='raw_mode')
    parser.add_argument('-scaling', '--scaling-type', type=str, nargs='?', default='unet',
                        help='scaling type?', dest='scaling_type')
    parser.add_argument('-test', '--test-only', type=int, nargs='?', default=1,
                        help='eval testset only??', dest='test_only')
    parser.add_argument('-all', '--all-cells', type=int, nargs='?', default=0,
                        help='inference the model by all cells??', dest='all_cells')
    parser.add_argument('-term1', type=int, nargs='?', default=0,
                        help='inference model from term1?', dest='term1')
    parser.add_argument('-coteaching', type=int, nargs='?', default=0,
                        help='inference co_teaching model?', dest='coteaching')
    parser.add_argument('-ensemble', type=int, nargs='?', default=0,
                        help='inference co_teaching model?', dest='ensem')
    parser.add_argument('-c2', '--checkpoint2', metavar='C', type=str, nargs='?', default=None,
                        help='the path of segmenter', dest='checkpoint2')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_num}'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    checkPoint = torch.load( args.checkpoint, map_location=device )
    net_2 = None
    
    if args.raw_mode:
        if args.coteaching:
            # load U-Net
            net = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net.load_state_dict( checkPoint['best_net1'] )
            #net.load_state_dict( checkPoint )
            net.to(device=device)
            net.eval()
            net_2 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net_2.load_state_dict( checkPoint['best_net2'] )
            net_2.to(device=device)
            net_2.eval()
        
        elif args.ensem:
            net = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net.load_state_dict( checkPoint['best_net'] )
            net.to(device=device)
            net.eval()
            checkPoint2 = torch.load( args.checkpoint2, map_location=device )
            net_2 = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net_2.load_state_dict( checkPoint2['best_net'] )
            net_2.to(device=device)
            net_2.eval()

        else:
            # load U-Net
            net = UNet(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            #print( checkPoint.keys() )
            net.load_state_dict( checkPoint['best_net'] )
            #net.load_state_dict( checkPoint )
            net.to(device=device)
            net.eval()
            net_2 = None

        net_g=None; net_s1=None; net_s2=None

    else:
        # load MCD-U-Net
        if args.term1:
            net_g = Generator(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net_g.load_state_dict( checkPoint['best_g_main'] )
            net_g.to(device=device)
            net_g.eval()
            net_s1 = Segmenter(first_num_of_kernels=args.first_num_of_kernels, n_channels=1, n_classes=1, bilinear=True)
            net_s1.load_state_dict( checkPoint['best_s_main'] )
            net_s1.to(device=device)
            net_s1.eval()
            
        else:
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

        net=None

    if args.test_only:
        # eval testset only
        testDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data/{args.cell}'
    else:
        # eval all train and val data
        testDir = f'/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/train_data/{args.cell}/cat_train'
    

    testFiles = sorted( glob.glob(f'{testDir}/*'), key=natural_keys )
    tests = create_testlist( testFiles, scaling_type=args.scaling_type )
    

    # create the result file
    dir_result = './infResult/eval_{}_fk{}'.format( args.out_dir, args.first_num_of_kernels )
    dir_imgs = f'{dir_result}/imgs'
    os.makedirs( dir_result, exist_ok=True )
    os.makedirs( dir_imgs, exist_ok=True )
    path_w = f'{dir_result}/evaluation.txt'


    if args.all_cells:
    # 全細胞種についてinference
        with open(path_w, mode='w') as f:
            f.write(f'inference all cells, model:{args.checkpoint}\n')

        cell_list = ['a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'shsy5y', 'skbr3', 'skov3']
        testDir = '/home/miyaki/unsupdomaada_for_semaseg_of_cell_images/LIVECell_dataset/test_data'

        for cell_name in cell_list:
            with open(path_w, mode='a') as f:
                f.write(f'----inference {cell_name}----\n')

            testFiles = sorted( glob.glob(f'{testDir}/{cell_name}/*'), key=natural_keys )
            tests = create_testlist( testFiles, scaling_type=args.scaling_type )

            # net_g=None; net_s1=None; net_s2=None
            IoU, precision, recall = eval_mcd( device, tests, model=net, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
            
            with open(path_w, mode='a') as f:
                #各精度に対応するファイル名を出力
                #for i, filename in enumerate(testFiles):
                #    f.write('{}:{}\n'.format( i, filename ))
                
                #f.write('Dice : {: .04f} +-{: .04f}\n'.format(statistics.mean(Dice), statistics.stdev(Dice)))
                f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))

    else:
    # 指定した細胞種のみinference
        with open(path_w, mode='w') as f:
            f.write(f'inference single cell({args.cell}, \n model:{args.checkpoint})\n')
        testFiles = sorted( glob.glob(f'{testDir}/*'), key=natural_keys )
        tests = create_testlist( testFiles, scaling_type=args.scaling_type )
        
        if args.raw_mode:
            if args.term1:
                # generator & segmenter from term2
                IoU = eval_mcd( device, tests, model=net, net_g=net_g, net_s=net_s1, net_s_another=None, raw=args.raw_mode , logfilePath=path_w)
            else:
                # normal MCD
                IoU, precision, recall = eval_mcd( device, tests, model=net, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('----inference net_1----\n')
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))

        else:
            # normal unet or co-teaching model
            if args.coteaching:
                # net1
                IoU, precision, recall = eval_mcd( device, tests, model=net, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('----inference net_1----\n')
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))

                # net2
                IoU, precision, recall = eval_mcd( device, tests, model=net_2, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('----inference net_2----\n')
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))

                # ensemble of net1 & net2
                IoU, precision, recall = eval_mcd( device, tests, model=net, model_2=net_2, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('----inference emsemble net_1_2----\n')
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))
            
            elif args.ensem:
                # args.checlpoint2 necessary

                # net1
                IoU, precision, recall = eval_mcd( device, tests, model=net, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('----inference net_1----\n')
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))

                # net2
                IoU, precision, recall = eval_mcd( device, tests, model=net_2, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('----inference net_2----\n')
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))
                    
                # ensemble of net1 & net2
                IoU, precision, recall = eval_mcd( device, tests, model=net, model_2=net_2, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('----inference emsemble net_1_2----\n')
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))

            else:
                # normal unet
                IoU, precision, recall = eval_mcd( device, tests, model=net, net_g=net_g, net_s=net_s1, net_s_another=net_s2, raw=args.raw_mode , logfilePath=path_w)
                with open(path_w, mode='a') as f:
                    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))
                    f.write('precision : {: .04f} +-{: .04f}\n'.format(statistics.mean(precision), statistics.stdev(precision)))
                    f.write('recall : {: .04f} +-{: .04f}\n'.format(statistics.mean(recall), statistics.stdev(recall)))
        
        #img_result, img_merge = segment(seg_shsy5y, net_g=net_g, net_s=net_s1, use_mcd=1)
        
        #with open(path_w, mode='a') as f:
        #    for i, filename in enumerate(testFiles):
        #        f.write('{}:{}\n'.format( i, filename ))
                
            #f.write('Dice : {: .04f} +-{: .04f}\n'.format(statistics.mean(Dice), statistics.stdev(Dice)))
        #    f.write('IoU : {: .04f} +-{: .04f}\n'.format(statistics.mean(IoU), statistics.stdev(IoU)))

        #io.imsave(f'{dir_imgs}/input.tif', imgSet[-2].reshape(img.shape[-2], img.shape[-1]))
        #io.imsave(f'{dir_imgs}/result.tif', img_result)
        #io.imsave(f'{dir_imgs}/merge.tif', img_merge)
