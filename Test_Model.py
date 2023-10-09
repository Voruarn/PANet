import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from network.PANet import PANet


if __name__ == "__main__":
    print('Test PANet !')
    """
    backbone_name: 
        resnet50, resnet101, resnet152
        hrnetv2_32, hrnetv2_48
        PoolFormer_S12, PoolFormer_S24, PoolFormer_S36, PoolFormer_S36,

    """
    backbone='resnet50'
    model_name='PANet'
    print('test {}_{}'.format(model_name, backbone))

    input=torch.rand(2,3,256,256).cuda()
    print('input.shape:',input.shape)
    
    n_classes=2

    model=PANet(n_channels=3, backbone_name=backbone).cuda()


    output=model(input)
    
    for out in output:
        print('out.shape:',out.shape)