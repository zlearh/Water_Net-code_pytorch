from utils import (
    imsave,
    prepare_data
)
import time
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch.nn as nn
import scipy.io as scio
import argparse
from model_train import *

parser = argparse.ArgumentParser()
# parser.add_argument('--exp_name', help='Where to store logs and models')
parser.add_argument('--train_data', default='./test_real/', help='path to training dataset')
parser.add_argument('--valid_data', default='', help='path to validation dataset')
parser.add_argument('--is_train_model', default='True', help='is train or test')
parser.add_argument('--save_model', default='True', help='continue train model')
parser.add_argument('--image_height', default=112, help='The size of image to use [230]')
parser.add_argument('--image_weight', default=112, help='The size of image to use [310]')
parser.add_argument('--label_height', default=112, help='The size of image to use [230]')
parser.add_argument('--label_weight', default=112, help='The size of image to use [310]')
parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch_num', type=int, default=500, help='epoch number')
parser.add_argument('--saved_model', default='', help="path to model to continue training")
parser.add_argument('--saved_model_vgg', default='/home/rtx3090/zhuang/water_net/data/images_2_该vgg损失/vgg_1.pth', help="path to model to continue training")
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=1.0 for Adadelta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.9')
opt = parser.parse_args()

""" Seed and GPU setting """
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

train(opt)