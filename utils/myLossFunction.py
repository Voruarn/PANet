import torch.nn.functional as F
import torch
import torch.nn as nn

import pytorch_ssim
import pytorch_iou
from .loss import FocalLoss


FOCAL_Loss = FocalLoss(ignore_index=255, size_average=True)
CSE_Loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
def multi_cse(d0,d1,d2,d3,d4, labels):
    loss0=CSE_Loss(d0, labels)
    loss1=CSE_Loss(d1, labels)
    loss2=CSE_Loss(d2, labels)
    loss3=CSE_Loss(d3, labels)
    loss4=CSE_Loss(d4, labels)
    loss=loss0 + loss1 + loss2/2 + loss3/4 + loss4/8
    return loss

def multi_focal(d0,d1,d2,d3,d4, labels):
    loss0=FOCAL_Loss(d0, labels)
    loss1=FOCAL_Loss(d1, labels)
    loss2=FOCAL_Loss(d2, labels)
    loss3=FOCAL_Loss(d3, labels)
    loss4=FOCAL_Loss(d4, labels)
    loss=loss0 + loss1 + loss2/2 + loss3/4 + loss4/8
    return loss


bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):
	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out + iou_out
    return loss

def bi_DS5(d0, d1, d2, d3, d4, labels_v):
    loss0 = bce_iou_loss(d0,labels_v)
    loss1 = bce_iou_loss(d1,labels_v)
    loss2 = bce_iou_loss(d1,labels_v)
    loss3 = bce_iou_loss(d1,labels_v)
    loss4 = bce_iou_loss(d1,labels_v)

    loss = loss0 + loss1 + loss2/2 + loss3/4 + loss4/8 
    return loss


def bsi_DS5(d0, d1, d2, d3, d4, labels_v):
    loss0 = bce_ssim_loss(d0,labels_v)
    loss1 = bce_ssim_loss(d1,labels_v)
    loss2 = bce_ssim_loss(d2,labels_v)
    loss3 = bce_ssim_loss(d3,labels_v)
    loss4 = bce_ssim_loss(d4,labels_v)

    loss = loss0 + loss1 + loss2/2 + loss3/4 + loss4/8 
    return loss

def bce_DS5(d0, d1, d2, d3, d4, labels_v):
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d1,labels_v)
    loss3 = bce_loss(d1,labels_v)
    loss4 = bce_loss(d1,labels_v)

    loss = loss0 + loss1 + loss2/2 + loss3/4 + loss4/8 
    return loss

def bce_DS4( d1, d2, d3, d4, labels_v):
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d1,labels_v)
    loss3 = bce_loss(d1,labels_v)
    loss4 = bce_loss(d1,labels_v)
    loss = loss1 + loss2/2 + loss3/4 + loss4/8 
    return loss

def bi_DS4(d1, d2, d3, d4, labels_v):
    loss1 = bce_iou_loss(d1,labels_v)
    loss2 = bce_iou_loss(d1,labels_v)
    loss3 = bce_iou_loss(d1,labels_v)
    loss4 = bce_iou_loss(d1,labels_v)
    loss = loss1 + loss2/2 + loss3/4 + loss4/8 
    return loss

def bsi_DS4(d1, d2, d3, d4, labels_v):
    loss1 = bce_ssim_loss(d1,labels_v)
    loss2 = bce_ssim_loss(d1,labels_v)
    loss3 = bce_ssim_loss(d1,labels_v)
    loss4 = bce_ssim_loss(d1,labels_v)
    loss = loss1 + loss2/2 + loss3/4 + loss4/8 
    return loss