# Pytorch-Water-Net

This is the code of the implementation of the underwater image enhancement network (Water-Net) described in "Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , IEEE TIP 2019"

# Requirement 

			Pytorch >= 1.4
      	Cuda  8.0, and Matlab
      torchvision >= 1.1.0
      random
      numpy

# Usage

# Test

1.Generate the preprocessing data by using the "generate_test_data.m" in folder named generate_test_data (Also, there is a modified code that includes WB, HE and GC in Python code without a need for preprocessing by MATLAB.  You can find the modified code in folder named testing_code_by_Branimir Ambrekovic. More details can be found in B's codes.)
Put the inputs to corresponding folders (raw images to "test_real", WB images to "wb_real", GC images to "gc_real", HE images to "ce_real")You can extract the file to get

vgg weight https://pan.baidu.com/s/1YeayV3xuwNWjXw6QX1tkew code：8cfn

2.Python main_.py

3.Python main_test.py

4.Find the result in file"data"

5.python measure_ssim_psnr.py

Set the network parameters, including learning rate, batch, weights of losses, etc., according to the paper
Generate the preprocessing training data by using the "generate_training_data.m" in folder named generate_test_data
Put the training data to corresponding folders (raw images to "input_train", WB images to "input_wb_train", GC images to "input_gc_train", HE images to "input_ce_train", Ground Truth images to "gt_train"); We randomly select the training data from our released dataset. The performance of different training data is almost same
In this code, you can add validation data by preprocessing your validation data (with GT) by the "generate_validation_data.m" in folder named generate_test_data, then put them to the corresponding folders (raw images to "input_test", WB images to "input_wb_test", GC images to "input_gc_test", HE images to "input_ce_test", Ground Truth images to "gt_test")
For your convenience, we provide a set of training and testing data. You can find them by unziping "a set of training and testing data". However, the training data and testing data are diffrent from those used in our paper.



