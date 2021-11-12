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
import numpy as np
import torch.nn as nn
from vgg import *
from torch.autograd import Variable
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):

    data_train_list = prepare_data(dataset="/home/rtx3090/zhuang/water_net/input_train/")
    data_wb_train_list = prepare_data(dataset="/home/rtx3090/zhuang/water_net/input_wb_train/")
    data_ce_train_list = prepare_data(dataset="/home/rtx3090/zhuang/water_net/input_ce_train/")
    data_gc_train_list = prepare_data(dataset="/home/rtx3090/zhuang/water_net/input_gc_train/")
    image_train_list = prepare_data(dataset="/home/rtx3090/zhuang/water_net/gt_train/")

    seed = 568
    np.random.seed(seed)
    np.random.shuffle(data_train_list)
    np.random.seed(seed)
    np.random.shuffle(data_wb_train_list)
    np.random.seed(seed)
    np.random.shuffle(data_ce_train_list)
    np.random.seed(seed)
    np.random.shuffle(data_gc_train_list)
    np.random.seed(seed)
    np.random.shuffle(image_train_list)
        
    sample_data_files = data_train_list
    sample_wb_data_files = data_wb_train_list
    sample_ce_data_files = data_ce_train_list
    sample_gc_data_files = data_gc_train_list
    sample_image_files = image_train_list
    sample_data,sample_wb_data_files_1,sample_ce_data_files_2,sample_gc_data_files_3 = list_to_tensor(sample_data_files,
                                                         sample_wb_data_files, sample_ce_data_files, sample_gc_data_files)
    sample_lable_image = [
          get_image(sample_image_file) for sample_image_file in sample_image_files]
    sample_lable_image = torch.Tensor(sample_lable_image).cuda().permute(0,3,1,2)
                                               
    sample_data,sample_wb_data_files,sample_ce_data_files,sample_gc_data_files=data_batch(sample_data,
                                                        sample_wb_data_files_1, sample_ce_data_files_2, sample_gc_data_files_3,opt.batch_size)
    sample_lable_image = torch.utils.data.DataLoader(dataset=sample_lable_image,
                                                 batch_size=opt.batch_size)
    model = Model()
    mode_vgg = VGG()

    if torch.cuda.is_available():
        model = model.cuda()
        mode_vgg = mode_vgg.cuda()
        Tensor = torch.cuda.FloatTensor


    if opt.saved_model != '':
        model.load_state_dict(torch.load(opt.saved_model))
    if opt.saved_model_vgg != '':
        mode_vgg.load_state_dict(torch.load(opt.saved_model_vgg))
    print("Model:")
    # Optimizers
    optimizer_V = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    #train vgg model
    #optimizer_Vgg = torch.optim.Adam(mode_vgg.parameters(), lr=0.001, betas=(0.9, 0.999))

    for epoch in range(opt.epoch_num):
        for (batch,batch_wb,batch_ce,batch_gc,batch_laber) in zip(sample_data,sample_wb_data_files,sample_ce_data_files,sample_gc_data_files,sample_lable_image):
            imgs_distorted_train = torch.autograd.Variable(batch)
            sample_wb_data__train = torch.autograd.Variable(batch_wb)
            sample_ce_data_train = torch.autograd.Variable(batch_ce)
            sample_gc_data_train = torch.autograd.Variable(batch_gc)



        ## Train Model
            optimizer_V.zero_grad()
            # train vgg model
            # optimizer_Vgg.zero_grad()
            pred_h1 = model(imgs_distorted_train,sample_wb_data__train,sample_ce_data_train,sample_gc_data_train)
            enhanced_texture_vgg = mode_vgg(pred_h1)
            labels_texture_vgg = mode_vgg(batch_laber)
            loss_texture = torch.mean(torch.square(enhanced_texture_vgg -labels_texture_vgg)).to(device)

            loss_h1 =  torch.mean(torch.abs(batch_laber - pred_h1)).to(device)
            loss = 0.05*loss_texture + loss_h1
            # train vgg model
            # loss_texture.backward()
            # optimizer_Vgg.step()
            loss.backward()
            optimizer_V.step()

        if not epoch % 10:
            imgs = next(iter(sample_data))
            imgs_val = Variable(imgs.type(Tensor))
            imgs1 = next(iter(sample_wb_data_files))
            imgs_val1 = Variable(imgs1.type(Tensor))
            imgs2 = next(iter(sample_ce_data_files))
            imgs_val2 = Variable(imgs2.type(Tensor))
            imgs3 = next(iter(sample_gc_data_files))
            imgs_val3 = Variable(imgs3.type(Tensor))
            print(imgs_val.shape,imgs_val1.shape,imgs_val2.shape,imgs_val3.shape)
            imgs_gen = model(imgs_val,imgs_val1,imgs_val2,imgs_val3)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, "/home/rtx3090/zhuang/water_net/data/%s.png" % (epoch))


        if not epoch % 10:
            print("[Epoch %d/%d] [loss_texture: %.3f, loss_h1: %.5f, loss_total: %.3f]"
                              %(
                                epoch, opt.epoch_num,loss_texture, loss_h1, loss
                               )
            )


        if (epoch % 10 == 0):
            #save model weight
            torch.save(model.state_dict(), "/home/rtx3090/zhuang/water_net/data/model_%d.pth" % (epoch))
            # train vgg model
            #torch.save(mode_vgg.state_dict(), "/home/rtx3090/zhuang/water_net/data/vgg_model_%d.pth" % (epoch))

def Conv2d(input,output,k_size,d_size):
    padding = (k_size-1)//2
    layer = [nn.Conv2d(input,output,k_size,d_size,padding),
             nn.ReLU(inplace=True)]
    return nn.Sequential(*layer)

def Tree_Conv2d():
    layer = [
        nn.Conv2d(6, 32, 7, 1,3),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 5, 1,2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 3, 3, 1,1),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layer)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.output_channel = [3,12,32,64,128]
        self.Conv2d_1 = Conv2d(self.output_channel[1],self.output_channel[4],7,1)
        self.Conv2d_2 = Conv2d(self.output_channel[4], self.output_channel[4], 5, 1)
        self.Conv2d_3 = Conv2d(self.output_channel[4], self.output_channel[4], 3, 1)
        self.Conv2d_4 = Conv2d(self.output_channel[4], self.output_channel[3], 1, 1)
        self.Conv2d_5 = Conv2d(self.output_channel[3], self.output_channel[3], 7, 1)
        self.Conv2d_6 = Conv2d(self.output_channel[3], self.output_channel[3], 5, 1)
        self.Conv2d_7 = Conv2d(self.output_channel[3], self.output_channel[3], 3, 1)
        self.Conv2d_8 = Conv2d(self.output_channel[3], 9, 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.Tree_Conv = Tree_Conv2d()

    def forward(self,images,images_wb,images_ce,images_gc):
        x = torch.cat((images,images_wb,images_ce,images_gc),1)
        x = self.Conv2d_1(x)
        x = self.Conv2d_2(x)
        x = self.Conv2d_3(x)
        x = self.Conv2d_4(x)
        x = self.Conv2d_5(x)
        x = self.Conv2d_6(x)
        x = self.Conv2d_7(x)
        x = self.Conv2d_8(x)
        x_1 = self.sigmoid(x)

        x_images_wb = torch.cat((images,images_wb),1)
        x_images_wb = self.Tree_Conv(x_images_wb)

        x_images_ce = torch.cat((images,images_ce),1)
        x_images_ce = self.Tree_Conv(x_images_ce)

        x_images_gc = torch.cat((images,images_gc),1)
        x_images_gc = self.Tree_Conv(x_images_gc)

        weight_wb,weight_ce,weight_gc = torch.split(x_1,3,1)
        x = torch.add(torch.add(torch.mul(x_images_wb,weight_wb),torch.mul(x_images_ce,weight_ce)),torch.mul(x_images_gc,weight_gc))
        return x
