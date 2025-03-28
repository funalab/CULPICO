import os
import re
import glob
import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import skimage.io
import skimage.transform
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tvf


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target, device):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda(device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class IOULoss(Function):
    
    def forward( self, input, target ):
        self.save_for_backward( input, target )
        eps = 0.0001
        input = input.view(-1)
        target = target.view(-1)
        self.intersection = torch.sum( input * target )
        self.total = torch.sum( input + target )
        self.union = self.total - self.intersection
        t = ( self.intersection.float() + eps ) / ( self.union.float() + eps )
                
        return t

def iou_loss( input, target, device ):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda(device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    
    for i, c in enumerate(zip(input, target)):
        s = s + IOULoss().forward(c[0], c[1])
    
    return s / (i + 1)
    
def random_rotate_image(image: np.ndarray, return_angle: bool = False, spin: int = None, flip: bool = None):
    """
    imageを4方向のうちランダムに回転する。
    +上下反転
    :param image: 元画像
    :param return_angle: Trueなら、回転角度を返り値に含む。
    :return: 回転後画像
    """

    image = image.copy()
    if spin != None:
        angle = spin
    else: 
        angle = random.choice((0, 90, 180, 270))

    new_img = skimage.transform.rotate(image, angle=angle, preserve_range=True).astype(image.dtype)
    
    if flip == None:
        flip = random.choice((True, False))
    
    if flip:
        new_img = np.flipud(new_img)
    
    if return_angle:
        return new_img, angle, flip
    else:
        return new_img

def batch_ver2(iterable, batch_size, cell):
    b = []
    for i, t in enumerate(iterable):
        ## ndim == 3の場合　t[i][0], t[i][1]　を　(1,128,128)->(128,128)
        #　random rotate (t[i][0] と　t[i][1] でangle 揃える)
        # -> reshape(1,128,128) 後　append
        #ndim == 2 を想定
        tmp_list = [0] * 2
        if t[0].ndim==2:
            rotate_img, ang, fl = random_rotate_image(t[0], return_angle=True)
            rotate_mask = random_rotate_image(t[1], spin=ang, flip=fl)
            if cell == 'bt474' or 'shsy5y':
                tmp_list = random_cropping( rotate_img, rotate_mask, 272, 352 )
                tmp_list[0] = tmp_list[0].reshape([1, tmp_list[0].shape[-2], tmp_list[0].shape[-1]])
                tmp_list[1] = tmp_list[1].reshape([1, tmp_list[1].shape[-2], tmp_list[1].shape[-1]])
            else:
                tmp_list[0] = rotate_img.reshape([1, rotate_img.shape[-2], rotate_img.shape[-1]])
                tmp_list[1] = rotate_mask.reshape([1, rotate_mask.shape[-2], rotate_mask.shape[-1]])
        else:
            print('ndim error!')
        
        b.append(tmp_list)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b
    
def batch(iterable, batch_size, source):
    
    b = []
    for i, t in enumerate(iterable):
        ## ndim == 3の場合　t[i][0], t[i][1]　を　(1,128,128)->(128,128)
        #　random rotate (t[i][0] と　t[i][1] でangle 揃える)
        # -> reshape(1,128,128) 後　append
        #ndim == 2 を想定
        tmp_list = [0] * 2
        if t[0].ndim==2:
            if source == 'HeLa':
                rotate_img, ang, fl = random_rotate_image(t[0], return_angle=True)
                rotate_mask = random_rotate_image(t[1], spin=ang, flip=fl)
                tmp_list[0] = rotate_img.reshape([1, rotate_img.shape[-2], rotate_img.shape[-1]])
                tmp_list[1] = rotate_mask.reshape([1, rotate_mask.shape[-2], rotate_mask.shape[-1]])
            
            else:
                tmp_list[0] = t[0].reshape([1, t[0].shape[-2], t[0].shape[-1]])
                tmp_list[1] = t[1].reshape([1, t[1].shape[-2], t[1].shape[-1]])
        else:
            print('ndim error!')
        
        b.append(tmp_list)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def batch_for_resnet(iterable, batch_size):
    b = []
    for i, t in enumerate(iterable):
        tmp_list = [0] * 2
        tmp_list[0] = np.transpose(t[0], (2, 0, 1))
        tmp_list[1] = t[1].reshape([1, t[1].shape[-2], t[1].shape[-1]])

        b.append(tmp_list)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def scaling_image(image):
    """
    画像の輝度値を0~1に正規化する
    :param image: 元画像
    :return image: 正規化後の画像
    """
    image = image.astype(np.float32)

    try:
        with np.errstate(all='raise'):
            image = (image - image.min()) / (image.max() - image.min())
    except (ZeroDivisionError, FloatingPointError):
        print("error!")

    return image

class Diff2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Diff2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        #predict the input already torch.sigmoided
        #torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))
        return torch.mean(torch.abs(inputs1 - inputs2))

def draw_graph( save_dir, graph_name, epochs, red_list=None, red_label=None, blue_list=None, blue_label=None,  green_list=None, green_label=None, x_label='epoch', y_label='loss' ):
    y_max = -10000
    graph = plt.figure()
    if red_list != None:
        plt.plot(range(epochs), red_list, 'r-', label=red_label )
        if y_max < max(red_list):
            y_max = max(red_list)
    if blue_list != None:
        plt.plot(range(epochs), blue_list, 'b-', label=blue_label )
        if y_max < max(blue_list):
            y_max = max(blue_list)
    if green_list != None:
        plt.plot(range(epochs), green_list, 'g-', label=green_label )
        if y_max < max(green_list):
            y_max = max(green_list)
    plt.legend()

        # 0.5間隔でrange設定
    ylim_max = math.ceil(y_max)-0.5 if math.ceil(y_max) - y_max > 0.5 else math.ceil(y_max)
    plt.ylim(0, ylim_max)

    plt.xlabel( x_label )
    plt.ylabel( y_label )
    plt.grid()
        
    graph.savefig('{}/{}.pdf'.format(save_dir, graph_name))
    plt.clf()
    plt.close()

def get_img_list(name, cell, large_flag):
    
    trains = []
    if large_flag:
        absolute = os.path.abspath(f'../../dataset_smiyaki/training_data/{cell}_raw')
        train_files = glob.glob(f"{absolute}/*")

        
    else:
        absolute = os.path.abspath('../../dataset_smiyaki')
        train_files = glob.glob(f"{absolute}/training_data/{cell}_set/*")
        
    for trainfile in train_files:
        ph_lab = [0] * 2
        #*set*/
        path_phase_and_lab = glob.glob(f"{trainfile}/*")
        #print(f"{trainfile}")
        for path_img in path_phase_and_lab:
            #print("hoge")
            img = io.imread(path_img)
            if name in path_img:
                #original unet scaling (subtracting by median)
                img = scaling_image(img)
                if large_flag:
                    img = img - np.median(img)

                #ndim==2でlistに格納
                ph_lab[0] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])

        trains.append(ph_lab)
    return trains

def mirror_padding( img, h, w ):

    ### size check ###
    if img.shape[0] >= h or img.shape[1] > w:
        print( "img is equal to or larger than specified size" ) 
        return img

    ### mirror padding process ###
    img_h = img.shape[0]
    img_w = img.shape[1]

    #print(f"img_h:{img_h}")
    #print(f"img_h:{img_w}")
    
    append_ha = np.empty( ( 0, img_w ), int )
    append_hb = np.empty( ( 0, img_w ), int )

    
    diff_h = h - img_h
    ha = int( diff_h / 2 )
    hb = int( diff_h / 2 ) if diff_h % 2 == 0 else int( diff_h / 2 ) + 1
    
    for i in range( ha ):
        append_ha = np.append( append_ha, img[ha-1-i].reshape( 1, img_w ), axis=0 )
    for i in range( hb ):
        append_hb = np.append( append_hb, img[img_h-1-i].reshape( 1, img_w ), axis=0 )

    #append above & below      
    img = np.append(append_ha, img, axis=0)
    img = np.append(img, append_hb, axis=0)

    diff_w = w - img_w
        
    wl = int( diff_w / 2 )

    if diff_w != 0:
        
        append_wl = np.empty( ( 0, wl ), int )
    
        for i in range( h ):
            append_wl = np.append( append_wl, img[i][wl-1::-1].reshape( 1, wl ), axis=0 )
                
    
        wr = wl if diff_w % 2 == 0 else wl+1
   
        append_wr = np.empty( ( 0, wr ), int )        
            
        for i in range( h ):
        
            append_wr = np.append( append_wr, img[i][img_w-1:img_w-wr-1:-1].reshape( 1, wr ), axis=0 )

        #append left & right                
        img = np.append(append_wl, img, axis=1)
        img = np.append(img, append_wr, axis=1) 
        
    return img

def random_cropping( img, lab, outH, outW ):

    #img, labはnp.ndarrayを想定
    rotate_img, ang, fl = random_rotate_image(img, return_angle=True)
    rotate_lab = random_rotate_image(lab, spin=ang, flip=fl)
    
    #trans = transforms.RandomCrop( ( outH, outW ) )
    
    pil_img = Image.fromarray( rotate_img )
    pil_lab = Image.fromarray( rotate_lab )

    i, j, h, w = transforms.RandomCrop.get_params( pil_img, output_size=( outH, outW ) )

    crop_img = np.array( tvf.crop( pil_img, i, j, h, w ) )

    crop_lab = np.array( tvf.crop( pil_lab, i, j, h, w ) )

    crop_list = [0] * 2
    
    crop_list[0] = crop_img
    crop_list[1] = crop_lab

    return crop_list


def random_cropping_for_resnet(img, lab, outH, outW):
    # img, labはnp.ndarrayを想定
    rotate_img, ang, fl = random_rotate_image(img, return_angle=True)
    rotate_lab = random_rotate_image(lab, spin=ang, flip=fl)

    pil_img = Image.fromarray( rotate_img )
    pil_lab = Image.fromarray( rotate_lab )

    i, j, h, w = transforms.RandomCrop.get_params( pil_img, output_size=( outH, outW ) )

    crop_img = np.array( tvf.crop( pil_img, i, j, h, w ) )
    crop_lab = np.array( tvf.crop( pil_lab, i, j, h, w ) )

    # to adjust for pretrained resnet
    crop_img = np.repeat(crop_img[..., np.newaxis], 3, axis=-1)

    #crop_img = np.transpose(crop_img, (2, 0, 1))
    #crop_lab = np.transpose(crop_lab, (2, 0, 1))

    crop_list = [0] * 2

    crop_list[0] = crop_img
    crop_list[1] = crop_lab

    return crop_list


def cutting_img( img_list, size_h, size_w ):
    height = img_list[0].shape[0]
    width = img_list[0].shape[1]
    #print(f"height={height}")
    #print(f"width={width}")
    n_h = int( height / size_h )
    n_w = int( width / size_w )
    #print(f"n_h={n_h}")
    #print(f"n_w={n_w}")
    #print(f"img_list[0].shape={img_list[0].shape}")
    #print(f"img_list[1].shape={img_list[1].shape}")

    cut_list = []
    cut_imgs = [0] * 2
    for i in range ( n_h ):
        for j in range ( n_w ):
            
            cut_imgs[0] = img_list[0][ i * size_h : i * size_h + size_h , j * size_w : j * size_w + size_w ]            
            cut_imgs[1] = img_list[1][ i * size_h : i * size_h + size_h , j * size_w : j * size_w + size_w ] 
            
            cut_list.append( copy.copy(cut_imgs) )
            
    return cut_list


def cutting_img_for_resnet(img_list, size_h, size_w):
    height = img_list[0].shape[0]
    width = img_list[0].shape[1]
    # print(f"height={height}")
    # print(f"width={width}")
    n_h = int(height / size_h)
    n_w = int(width / size_w)
    # print(f"n_h={n_h}")
    # print(f"n_w={n_w}")
    # print(f"img_list[0].shape={img_list[0].shape}")
    # print(f"img_list[1].shape={img_list[1].shape}")

    cut_list = []
    cut_imgs = [0] * 2
    for i in range(n_h):
        for j in range(n_w):
            img_cut = img_list[0][i * size_h: i * size_h + size_h, j * size_w: j * size_w + size_w]
            # to adjust for pretrained resnet
            cut_imgs[0] = np.repeat(img_cut[..., np.newaxis], 3, axis=-1)
            cut_imgs[1] = img_list[1][i * size_h: i * size_h + size_h, j * size_w: j * size_w + size_w]

            cut_list.append(copy.copy(cut_imgs))

    return cut_list


def reshape_for_validation_in_resnet(img, mask, device):
    img = np.squeeze(img, axis=0)
    img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    img = np.array(img).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(device=device)

    mask = np.array(mask).astype(np.float32)
    mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
    mask = torch.from_numpy(mask).unsqueeze(0).to(device=device)

    return img, mask.shape[-2:]


def reshape_for_validation_in_dual(img, mask, device):
    img_t = np.array(img).astype(np.float32)
    img_t = torch.from_numpy(img_t).unsqueeze(0).to(device=device)
    ###
    mask = np.array(mask).astype(np.float32)
    mask = mask.reshape([1, mask.shape[-2], mask.shape[-1]])
    mask = torch.from_numpy(mask).unsqueeze(0).to(device=device)
    return img_t, mask.shape[2:]


def adjust_img( img, height, width ):
    img_h = img.shape[0]
    img_w = img.shape[-1]
    
    diff_h = int( ( img_h - height ) / 2 )
    diff_w = int( ( img_w - width ) / 2 )

    new_img = img[ diff_h : diff_h + height, diff_w : diff_w + width ]

    return new_img

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



def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, device):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    SMOOTH = 1e-6

    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    iou_outputs = torch.where( outputs > 0.5, torch.tensor(1, dtype=torch.uint8, device=torch.device(device)), \
                                            torch.tensor(0, dtype=torch.uint8, device=torch.device(device)) )
    
    iou_labels = labels.to(torch.uint8)
    

    intersection = (iou_outputs & iou_labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (iou_outputs | iou_labels).float().sum()         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    #return thresholded  # Or thresholded.mean() if you are interested in average across the batch

    return iou

def precision_recall_pytorch(outputs: torch.Tensor, labels: torch.Tensor, device):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    SMOOTH = 1e-6
    
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    pr_outputs = torch.where( outputs > 0.5, torch.tensor(1, dtype=torch.uint8, device=torch.device(device)), \
                                            torch.tensor(0, dtype=torch.uint8, device=torch.device(device)) )
    
    pr_labels = labels.to(torch.uint8)
    
    TP = ( (pr_outputs == 1) & (pr_outputs == pr_labels) ).float().sum()
    FP = ( (pr_outputs == 1) & (pr_outputs != pr_labels) ).float().sum()
    FN = ( (pr_outputs == 0) & (pr_outputs != pr_labels) ).float().sum()
    
    precision = ( TP + SMOOTH ) / ( TP + FP + SMOOTH )
    
    recall = ( TP + SMOOTH ) / ( TP + FN + SMOOTH )

    
    return precision, recall


def iou_numpy(outputs: np.array, labels: np.array):

    SMOOTH = 1e-6

    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

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

def create_pseudo_label( p1, p2, T_dis, T_object=0.5, conf=0, device='cpu' ):
    # p1:output of S1(after sigmoid), p2:output of S2
    # T_dis: discrepancy threshold
    # conf: confidence of pseudo label 
    p_mean = ( p1 + p2 ) / 2
    p_conf = torch.abs( p_mean - T_object )
    dis = torch.abs( p1 - p2 )
    # 条件を満たすpixelのindex: decide(tuple)
    decide = torch.where( (dis<T_dis) & (p_conf>conf) )
    # 対象pixel(decide)にpseudo label( 1 or 0 )付与
    pseudo_lab = torch.where( p1[decide] > 0.5, torch.tensor(1, dtype=dis.dtype, device=torch.device(device)), torch.tensor(0, dtype=dis.dtype, device=torch.device(device)) )
    
    """
    tmp_label = torch.where( p_mean>T_object, torch.tensor(1, dtype=dis.dtype, device=torch.device(device)), torch.tensor(0, dtype=dis.dtype, device=torch.device(device)) )
    pselab_p1 = torch.where( dis<T_dis, tmp_label, p1 )
    pselab_p1 = torch.where( p_conf > conf, pselab_p1, p1 )
    pselab_p2 = torch.where( dis<T_dis, tmp_label, p2 )
    pselab_p2 = torch.where( p_conf > conf, pselab_p2, p2 )
    loss_dis = torch.mean( dis[torch.where( dis>=T_dis )] )
    """
    assigned = torch.numel(p1[decide])/torch.numel(p1)

    return decide, pseudo_lab, assigned

def standardize_image(img: np.ndarray):
    """
    画像を標準化する。(z-score を求める)
    :param img: 元画像。
    :return: 標準化後の画像
    """
    img = img.copy()
    img = img.astype(np.float32)
    img = (img - img.mean()) / img.std()

    return img



def create_trainlist(setList, scaling_type='unet', test=False, cut=False):

    trains = []
    max_intensity = 0
    min_intensity = 2 ** 16 - 1
    
    for setPath in setList:
        imgSet = [0] * 2
        
        filepathList = glob.glob(f"{setPath}/*")
        
        for filePath in filepathList:
            
            img = io.imread( filePath )
            if 'Phase' in filePath:
                max_intensity = max(max_intensity, img.max())
                min_intensity = min(min_intensity, img.min())
                
                if scaling_type == "unet":
                    img = scaling_image(img)
                    img = img - np.median(img)
                elif scaling_type == "standard":
                    img = standardize_image(img)
                elif scaling_type == "normal":
                    img = scaling_image(img)

                if scaling_type != 'all':
                    if cut: img = img[130:390, 176:528]
                imgSet[-2] = img if test == False else img.reshape([1, img.shape[-2], img.shape[-1]])
                    
            elif 'Mask' in filePath:
                img = img / 255
                if cut == True: img = img[130:390, 176:528]
                imgSet[-1] = img if test == False else img.reshape([1, img.shape[-2], img.shape[-1]])
        trains.append(imgSet)

        if scaling_type == 'all':
            for train in trains:
                train[-2] = train[-2].astype(np.float32)
                try:
                    with np.errstate(all='raise'):
                        train[-2] = (train[-2] - min_intensity) / (max_intensity - min_intensity)
                except (ZeroDivisionError, FloatingPointError):
                    pass
        
    return trains

def create_testlist(setList, scaling_type='unet'):
    # specified for inferencing 520x704 by 272x352
    # scalingtype = "all" is forbidden
    tests = []
    max_intensity = 0
    min_intensity = 2 ** 16 - 1
    
    for setPath in setList:
        imgSet = [[0] * 2 for i in range(4)]
        
        filepathList = glob.glob(f"{setPath}/*")
        
        for filePath in filepathList:
            
            img = io.imread( filePath )
            if 'Phase' in filePath:
                max_intensity = max(max_intensity, img.max())
                min_intensity = min(min_intensity, img.min())
                
                if scaling_type == "unet":
                    img = scaling_image(img)
                    img = img - np.median(img)
                elif scaling_type == "standard":
                    img = standardize_image(img)
                elif scaling_type == "normal":
                    img = scaling_image(img)
                
                imgSet[0][0] = img[0:272, 0:352].reshape([1, 272, 352])
                imgSet[1][0] = img[0:272, 352:704].reshape([1, 272, 352])
                imgSet[2][0] = img[248:520, 0:352].reshape([1, 272, 352])
                imgSet[3][0] = img[248:520, 352:704].reshape([1, 272, 352])
                
            elif 'Mask' in filePath:
                img = img / 255
                imgSet[0][1] = img[0:272, 0:352].reshape([1, 272, 352])
                imgSet[1][1] = img[0:272, 352:704].reshape([1, 272, 352])
                imgSet[2][1] = img[248:520, 0:352].reshape([1, 272, 352])
                imgSet[3][1] = img[248:520, 352:704].reshape([1, 272, 352])

        tests.append(imgSet)

    return tests

def create_uncer_pseudo( p1, p2, device='cpu' ):

    p_dis = torch.abs( p1 - p2 ) 

    pseudo_lab_p1 = torch.where( p1 > 0.5, torch.tensor(1, dtype=p_dis.dtype, device=torch.device(device)), torch.tensor(0, dtype=p_dis.dtype, device=torch.device(device)) )

    pseudo_lab_p2 = torch.where( p2 > 0.5, torch.tensor(1, dtype=p_dis.dtype, device=torch.device(device)), torch.tensor(0, dtype=p_dis.dtype, device=torch.device(device)) )

    confidence = 1 - p_dis

    return pseudo_lab_p1, pseudo_lab_p2, confidence

def loss_coteaching(y_1, y_2, t, forget_rate, batch_size, device):
    # y: pred(after sigmoid) before flatten
    # t: mask before flatten
    criterion_nored = nn.BCELoss(reduction='none')
    criterion = nn.BCELoss()
    
    # reshape
    # size(batch, 1, H, W) -> size(batch, H x W)
    size = t.size()
    y_1_inst_flat = torch.reshape( y_1, (batch_size, size[-1]*size[-2]) )
    y_2_inst_flat = torch.reshape( y_2, (batch_size, size[-1]*size[-2]) )
    t_inst_flat = torch.reshape( t, (batch_size, size[-1]*size[-2]) )
    
    # calc loss for each instance in batch
    loss_1 = torch.mean( criterion_nored( y_1_inst_flat, t_inst_flat ), dim = 1)
    ind_1_sorted = torch.argsort(loss_1.data)
    #np.argsort().cuda(device)
    loss_1_sorted = loss_1[ind_1_sorted]
    
    loss_2 = torch.mean( criterion_nored( y_2_inst_flat, t_inst_flat ), dim = 1)
    ind_2_sorted = torch.argsort(loss_2.data)
    #np.argsort().cuda(device)
    loss_2_sorted = loss_2[ind_2_sorted]
    
    
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    
    # exchange
    loss_1_update = criterion(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = criterion(y_2[ind_1_update], t[ind_1_update])
    
    #torch.sum(loss_1_update)/num_remember
    
    return loss_1_update, loss_2_update
