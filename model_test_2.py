from utils import (
    imsave,
    prepare_data
)
import random
import torch
import torch.backends.cudnn as cudnn
from utils import *
import sys
import time
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch.nn as nn
import scipy.io as scio
import argparse
from vgg import *
from torch.autograd import Variable
from model_train import Model
from PIL import Image
from torchvision.utils import save_image
from os.path import exists
from measure_ssim_psnr import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(opt):
    ## checks
    for name in range(0,10000,100):
        assert exists(opt.model_path), "model not found"
        model = Model()
        model_path = '/home/rtx3090/zhuang/water_net/data/model_' + str(j) + '.pth'
        Tensor = torch.FloatTensor
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Loaded model from model_%s sccees "%(name))
    
        data_test_lists = prepare_data(dataset=opt.data_test_list)
        data_wb_test_lists = prepare_data(dataset=opt.data_wb_test_list)
        data_ce_test_lists = prepare_data(dataset=opt.data_ce_test_list)
        data_gc_test_lists = prepare_data(dataset=opt.data_gc_test_list)
    
        ## testing loop
        data_test = sorted(data_test_lists)
        data_wb_test = sorted(data_wb_test_lists)
        data_ce_test = sorted(data_ce_test_lists)
        data_gc_test = sorted(data_gc_test_lists)
    
        data_test,data_wb_test,data_ce_test,data_gc_test = list_to_tensor(data_test,data_wb_test,data_ce_test,data_gc_test)
        data_test,data_wb_test,data_ce_test,data_gc_test = data_batch(data_test,data_wb_test,data_ce_test,data_gc_test,1)
        i = 0
        for (batch,batch_wb,batch_ce,batch_gc) in zip(data_test,data_wb_test,data_ce_test,data_gc_test):
            imgs_val = Variable(batch.type(Tensor))
            imgs_val1 = Variable(batch_wb.type(Tensor))
            imgs_val2 = Variable(batch_ce.type(Tensor))
            imgs_val3 = Variable(batch_gc.type(Tensor))
            print(imgs_val.shape, imgs_val1.shape, imgs_val2.shape, imgs_val3.shape)
            imgs_gen = model(imgs_val, imgs_val1, imgs_val2, imgs_val3)
            save_image(imgs_gen, "/home/rtx3090/zhuang/water_net/data/images_2_该vgg损失/re_images/%s.png" % (i))
            i = i+1
        print(" model from model_%s ssim_psnr"%(name))
        measure()

## options
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="water_net")
parser.add_argument("--data_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_test/")
parser.add_argument("--data_wb_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_wb_test/")
parser.add_argument("--data_ce_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_ce_test/")
parser.add_argument("--data_gc_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_gc_test/")
parser.add_argument("--save_images", type=str, default="/home/rtx3090/zhuang/water_net/data/images_2_该vgg损失/re_images/")
parser.add_argument("--model_path", type=str,default='/home/rtx3090/zhuang/water_net/data/model_2000.pth')
opt = parser.parse_args()

test(opt)