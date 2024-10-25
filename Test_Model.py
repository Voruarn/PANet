import torch
from torch.autograd import Variable
import torch.nn as nn
from network.PANet import PANet


if __name__ == "__main__":
    print('Test PANet !')

    input=torch.rand(2,3,512,512).cuda()
    print('input.shape:',input.shape)
    
    model=PANet().cuda()


    output=model(input)
    
    for out in output:
        print('out.shape:',out.shape)