import os
import glob
import scipy.misc
import imageio
import scipy.ndimage
import numpy as np
import torch
from torch.autograd import Variable

def transform(images):
    return np.array(images)/127.5-1.

def inverse_transform(images):
    return (images + 1.) / 2

def prepare_data(dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']

    """
    filenames = os.listdir(dataset)
    data_dir = os.path.join('C:/Users/zhuang/Desktop/water_net',dataset)
    data = glob.glob(os.path.join(data_dir,"*.png"))
    data = data + glob.glob(os.path.join(data_dir,"*.jpg"))
    return data

def imread(path,is_grayscale = False):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return imageio.imread(path,flatten = True).astype(np.float)  #灰度图
    else:
        return imageio.imread(path).astype(np.float)                 #彩色图

def imsave(image,path):
    imsaved = (inverse_transform(image)).astype(np.float)
    return imageio.imwrite(path,imsaved)

def get_image(image_path,is_grayscale=False):
    image = imread(image_path,is_grayscale)
    return image/255

def get_label(image_path,is_grayscale=False):
    image = imread(image_path,is_grayscale)
    return image/255.

def list_to_tensor(sample_data_files,sample_wb_data_files, sample_ce_data_files, sample_gc_data_files):
    sample_data = [
          get_image(sample_data_file) for sample_data_file in sample_data_files]

    sample_data = torch.Tensor(sample_data).cuda().permute(0,3,1,2)

    sample_wb_data_files_1 = [
          get_image(sample_image_file) for sample_image_file in sample_wb_data_files]
    sample_wb_data_files_1 = torch.Tensor(sample_wb_data_files_1).cuda().permute(0,3,1,2)

    sample_ce_data_files_2 = [
          get_image(sample_image_file) for sample_image_file in sample_ce_data_files]
    sample_ce_data_files_2 = torch.Tensor(sample_ce_data_files_2).cuda().permute(0,3,1,2)

    sample_gc_data_files_3 = [
          get_image(sample_image_file) for sample_image_file in sample_gc_data_files]
    sample_gc_data_files_3 = torch.Tensor(sample_gc_data_files_3).cuda().permute(0,3,1,2)
    return sample_data,sample_wb_data_files_1,sample_ce_data_files_2,sample_gc_data_files_3

def data_batch(sample_data,sample_wb_data_files_1,sample_ce_data_files_2,sample_gc_data_files_3,batch_size):
    sample_data = torch.utils.data.DataLoader(dataset=sample_data,
                                              batch_size=batch_size)

    sample_wb_data_files = torch.utils.data.DataLoader(dataset=sample_wb_data_files_1,
                                                       batch_size=batch_size,
                                                       )

    sample_ce_data_files = torch.utils.data.DataLoader(dataset=sample_ce_data_files_2,
                                                       batch_size=batch_size,
                                                       )
    sample_gc_data_files = torch.utils.data.DataLoader(dataset=sample_gc_data_files_3,
                                                       batch_size=batch_size,
                                                       )
    return sample_data,sample_wb_data_files,sample_ce_data_files,sample_gc_data_files


