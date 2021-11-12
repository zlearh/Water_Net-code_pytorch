import torch
import torch.backends.cudnn as cudnn
from utils import *
import numpy as np
import torch.nn as nn
import argparse
from vgg import *
from torch.autograd import Variable
from model_train import Model
from torchvision.utils import save_image
from os.path import exists

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(opt):
    ## checks
    assert exists(opt.model_path), "model not found"
    model = Model()

    Tensor = torch.FloatTensor
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    print("Loaded model from sccees")

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
    i = 1
    for (batch,batch_wb,batch_ce,batch_gc) in zip(data_test,data_wb_test,data_ce_test,data_gc_test):
        imgs_val = Variable(batch.type(Tensor))
        imgs_val1 = Variable(batch_wb.type(Tensor))
        imgs_val2 = Variable(batch_ce.type(Tensor))
        imgs_val3 = Variable(batch_gc.type(Tensor))
        imgs_gen = model(imgs_val, imgs_val1, imgs_val2, imgs_val3)
        save_image(imgs_gen, "/home/rtx3090/zhuang/water_net/data/images_2_该vgg损失/re_images/%s.png" % (i))
        i = i+1

## options
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="water_net")
parser.add_argument("--data_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_test/")
parser.add_argument("--data_wb_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_wb_test/")
parser.add_argument("--data_ce_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_ce_test/")
parser.add_argument("--data_gc_test_list", type=str, default="/home/rtx3090/zhuang/water_net/input_gc_test/")
parser.add_argument("--save_images", type=str, default="/home/rtx3090/zhuang/water_net/data/images_2_该vgg损失/re_images/")
parser.add_argument("--model_path", type=str,default='/home/rtx3090/zhuang/water_net/data/model_150.pth')
opt = parser.parse_args()

test(opt)