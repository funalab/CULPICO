#Diceloss
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage.io
import skimage.transform
import random
from skimage import io
import os
import glob
import matplotlib.pyplot as plt

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

def batch_t(iterable, batch_size):
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
            tmp_list[0] =   rotate_img.reshape([1, rotate_img.shape[-2], rotate_img.shape[-1]])
            tmp_list[1] =   rotate_mask.reshape([1, rotate_mask.shape[-2], rotate_mask.shape[-1]])
        else:
            print('ndim error!')
        
        b.append(tmp_list)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b
    
def batch(iterable, batch_size):
    
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
            tmp_list[0] =   rotate_img.reshape([1, rotate_img.shape[-2], rotate_img.shape[-1]])
            tmp_list[1] =   rotate_mask.reshape([1, rotate_mask.shape[-2], rotate_mask.shape[-1]])
        else:
            print('ndim error!')
        
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
        if image_name:
            print('empty image file : ' + str(image_name))

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
        graph = plt.figure()
        if red_list != None:
            plt.plot(range(epochs), red_list, 'r-', label=red_label )
        if blue_list != None:
            plt.plot(range(epochs), blue_list, 'b-', label=blue_label )
        if green_list != None:
            plt.plot(range(epochs), green_list, 'g-', label=green_label )
        plt.legend()
        plt.xlabel( x_label )
        plt.ylabel( y_label )
        plt.grid()
        
        graph.savefig('{}{}.pdf'.format(save_dir, graph_name))
        plt.clf()
        plt.close()

def get_img_list(name, cell):
    
    trains = []
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
                #img = img - np.median(img)

                #ndim==2でlistに格納
                ph_lab[0] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])
            else:
                img = img / 255
                ph_lab[1] = img
                #img.reshape([1, img.shape[-2], img.shape[-1]])

        trains.append(ph_lab)
    return trains
