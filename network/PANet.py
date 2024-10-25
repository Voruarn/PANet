import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .init_weights import init_weights
from .modules import *
from .ResNet import resnet18, resnet34, resnet50


class PANet(nn.Module):

    def __init__(self, encoder='resnet50', bottleneck_num=1, **kwargs):
        super(PANet, self).__init__()      

        
        eout_channels=[256, 512, 1024, 2048]
        if encoder in ['resnet18', 'resnet34']:
            eout_channels=[64, 128, 256, 512]
     
        self.backbone  = eval(encoder)(pretrained=False)

        n_classes=1
        mid_ch=64

        self.eside1=ConvModule(eout_channels[0], mid_ch)
        self.eside2=ConvModule(eout_channels[1], mid_ch)
        self.eside3=ConvModule(eout_channels[2], mid_ch)
        self.eside4=ConvModule(eout_channels[3], mid_ch)

        self.cbam=CBAM(mid_ch)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dblock1=DBlock(mid_ch, mid_ch, bottleneck_num)
        self.dblock2=DBlock(mid_ch, mid_ch, bottleneck_num)
        self.dblock3=DBlock(mid_ch, mid_ch, bottleneck_num)

        # Decoder
        self.CV1=ConvModule(mid_ch, mid_ch, 3, 2, 1)
        self.dblock4=DBlock(mid_ch, mid_ch, bottleneck_num)

        self.CV2=ConvModule(mid_ch, mid_ch, 3, 2, 1)
        self.dblock5=DBlock(mid_ch, mid_ch, bottleneck_num)

        self.CV3=ConvModule(mid_ch, mid_ch, 3, 2, 1)
        self.dblock6=DBlock(mid_ch, mid_ch, bottleneck_num)

        self.FFM1=FFM(mid_ch, mid_ch, mid_ch)
        self.FFM2=FFM(mid_ch, mid_ch, mid_ch)
        self.FFM3=FFM(mid_ch, mid_ch, mid_ch)

        self.dside1 = nn.Conv2d(mid_ch,n_classes,3,padding=1)
        self.dside2 = nn.Conv2d(mid_ch,n_classes,3,padding=1)
        self.dside3 = nn.Conv2d(mid_ch,n_classes,3,padding=1)
        self.dside4 = nn.Conv2d(mid_ch,n_classes,3,padding=1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # backbone, encoder
        outs = self.backbone(inputs)

        c1, c2, c3, c4 = outs

        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)
        dff=self.cbam(c1) # dff: detail feature flow
        dff1=dff
        dff2=F.interpolate(dff, size=c2.size()[2:], mode='bilinear', align_corners=False)
        dff3=F.interpolate(dff, size=c3.size()[2:], mode='bilinear', align_corners=False)

        # bottom-up PAD
        up3=self.upsample2(c4) + c3 + dff3
        up3=self.dblock3(up3)

        up2=self.upsample2(up3) + c2 + dff2
        up2=self.dblock2(up2)

        up1=self.upsample2(up2) + c1 + dff1
        up1=self.dblock1(up1)

        ## lff: location feature flow
        lff2 = F.interpolate(up1, size=up2.size()[2:], mode='bilinear')
        lff3 = F.interpolate(up1, size=up3.size()[2:], mode='bilinear')
        lff4 = F.interpolate(up1, size=c4.size()[2:], mode='bilinear')
    
        # top-down PAD
        dn2=self.CV1(up1) + up2 + lff2
        dn2=self.dblock4(dn2)

        dn3=self.CV2(dn2) + up3 + lff3
        dn3=self.dblock5(dn3)

        dn4=self.CV3(dn3) + c4 + lff4
        dn4=self.dblock6(dn4)

        dn3=self.FFM3(dn3, self.upsample2(dn4))
        dn2=self.FFM2(dn2, self.upsample2(dn3))
        dn1=self.FFM1(up1, self.upsample2(dn2))

        # side pred
        d1=self.dside1(dn1)
        d2=self.dside2(dn2)
        d3=self.dside3(dn3)
        d4=self.dside4(dn4)

        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)

        return S1,S2,S3,S4, torch.sigmoid(S1), torch.sigmoid(S2), torch.sigmoid(S3), torch.sigmoid(S4)
    