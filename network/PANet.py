import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .init_weights import init_weights
from .modules import *
from .ResNet import resnet50, resnet101, resnet152
from .poolformer_cus import PoolFormer
from .hrnetv2 import *
from .DualAttn import DualAttention

    
## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src=F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src

poolformer_settings = {
    'S12': [64, 128, 320, 512],   
    'S24': [64, 128, 320, 512],     # [layers, embed_dims, drop_path_rate]
    'S36': [64, 128, 320, 512],
    'M36': [96, 192, 384, 768]
}

HRNet_out_ch = {
    'w32': [32, 64, 128, 256],   
    'w48': [48, 96, 192, 384],     # [layers, embed_dims, drop_path_rate]
}

"""
backbone_name: 
    resnet50, resnet101, resnet152
    hrnetv2_48, hrnetv2_32
    PoolFormer_S12, PoolFormer_S24, PoolFormer_S36, PoolFormer_S36,

"""
class PANet(nn.Module):
    def __init__(self, n_channels=3, backbone_name='resnet50', bottleneck_num=2):
        super(PANet, self).__init__()      

        
        eout_channels=[256, 512, 1024, 2048]
        if backbone_name.find('hrnet')!=-1: # HRNet
            phi='w'+backbone_name.split('_')[-1]
            eout_channels=HRNet_out_ch[phi]
            self.backbone  = eval(backbone_name)(pretrained=False)
        elif backbone_name.find('PoolFormer')!=-1: # PoolFormer
            phi=backbone_name.split('_')[-1]
            eout_channels=poolformer_settings[phi]
            self.backbone=PoolFormer(phi)
        else:   # ResNet
            self.backbone  = eval(backbone_name)(pretrained=False)

        n_classes=1
        mid_ch=64

        self.eside1=ConvModule(eout_channels[0], mid_ch)
        self.eside2=ConvModule(eout_channels[1], mid_ch)
        self.eside3=ConvModule(eout_channels[2], mid_ch)
        self.eside4=ConvModule(eout_channels[3], mid_ch)

        self.DA=DualAttention(mid_ch, mid_ch)

        self.CSP1=C2(mid_ch*3, mid_ch, bottleneck_num)
        self.CSP2=C2(mid_ch*3, mid_ch, bottleneck_num)
        self.CSP3=C2(mid_ch*3, mid_ch, bottleneck_num)

        # DownCSP
        self.CV1=ConvModule(mid_ch, mid_ch, 3, 2, 1)
        self.CSP4=C2(mid_ch*2, mid_ch, bottleneck_num)

        self.CV2=ConvModule(mid_ch, mid_ch, 3, 2, 1)
        self.CSP5=C2(mid_ch*2, mid_ch, bottleneck_num)

        self.CV3=ConvModule(mid_ch, mid_ch, 3, 2, 1)
        self.CSP6=C2(mid_ch*2, mid_ch, bottleneck_num)

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
        gcf=self.DA(c4)

        # UpCSP
        up3=F.interpolate(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
        gcf3=F.interpolate(gcf, size=c3.size()[2:], mode='bilinear', align_corners=False)
        up3=torch.cat([c3,up3,gcf3], dim=1)
        up3=self.CSP3(up3)

        up2=F.interpolate(up3, size=c2.size()[2:], mode='bilinear', align_corners=False)
        gcf2=F.interpolate(gcf, size=c2.size()[2:], mode='bilinear', align_corners=False)
        up2=torch.cat([c2,up2,gcf2], dim=1)
        up2=self.CSP2(up2)

        up1=F.interpolate(up2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        gcf1=F.interpolate(gcf, size=c1.size()[2:], mode='bilinear', align_corners=False)
        up1=torch.cat([c1,up1,gcf1], dim=1)
        up1=self.CSP1(up1)

        fback2 = F.interpolate(up1, size=up2.size()[2:], mode='bilinear')
        fback3 = F.interpolate(up1, size=up3.size()[2:], mode='bilinear')
        fback4 = F.interpolate(up1, size=c4.size()[2:], mode='bilinear')
    
        # DownCSP
        dn2=self.CV1(up1)
        dn2=torch.cat([dn2, up2+fback2], dim=1)
        dn2=self.CSP4(dn2)

        dn3=self.CV2(dn2)
        dn3=torch.cat([dn3, up3+fback3], dim=1)
        dn3=self.CSP5(dn3)

        dn4=self.CV3(dn3)
        dn4=torch.cat([dn4, c4+fback4], dim=1)
        dn4=self.CSP6(dn4)

        dn3=self.FFM3(dn3, F.interpolate(dn4, size=dn3.size()[2:], mode='bilinear'))
        dn2=self.FFM2(dn2, F.interpolate(dn3, size=dn2.size()[2:], mode='bilinear'))
        dn1=self.FFM1(up1, F.interpolate(dn2, size=up1.size()[2:], mode='bilinear'))

        # side pred
        d1=self.dside1(dn1)
        d2=self.dside2(dn2)
        d3=self.dside3(dn3)
        d4=self.dside4(dn4)

        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)

        return torch.sigmoid(S1), torch.sigmoid(S2), torch.sigmoid(S3), torch.sigmoid(S4)
    