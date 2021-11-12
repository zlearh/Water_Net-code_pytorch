import torch
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG(nn.Module):
    def __init__(self, n_classes=3):
        super(VGG, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,16], [16,16], [3,3], [1,1], 1, 1)
        self.layer2 = vgg_conv_block([16,32], [32,32], [3,3], [1,1], 1, 1)
        self.layer3 = vgg_conv_block([32,64,64], [64,64,64], [7,7,7], [1,1,1], 1, 2)
        self.layer5 = vgg_conv_block([64,128,128], [128,128,128], [5,5,5], [1,1,1], 1, 2)
        self.layer6 = vgg_conv_block([128,128,128], [128,128,128], [3,3,3], [1,1,1], 1,2)
        self.layer7 = vgg_conv_block([128,256,256], [256,256,256], [7,7,7], [1,1,1], 1, 2)
        self.layer8 = vgg_conv_block([256,512,512], [512,512,512], [5,5,5], [1,1,1], 1, 2)
        self.layer9 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 1,2)

        self.layer10 = nn.Conv2d(128,3,1,1)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out1 = self.layer5(out)
        out2 = self.layer6(out1)
        out3 = self.layer10(out2)

        return out3

if __name__ == '__main__':
    # Example
    vgg = vgg()
    print(vgg19)