from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data
from datasets.EORSSD_Dataset import EORSSDDataset
from utils import ext_transforms as et
from metrics.SOD_metrics import SODMetrics
from network.PANet import PANet
import torch.nn.functional as F
import torch
import torch.nn as nn

from utils.myLossFunction import bce_DS4, bi_DS4


def get_argparser():
    parser = argparse.ArgumentParser()
 
    
    parser.add_argument("--trainset_path", type=str, 
        default='',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='EORSSD', help='Name of dataset:[EORSSD, ORSSD]')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
    """
    backbone_name: 
        resnet50, resnet101, resnet152
        hrnetv2_32, hrnetv2_48
        PoolFormer_S12, PoolFormer_S24, PoolFormer_S36, PoolFormer_S36,

    """
    parser.add_argument("--backbone", type=str, default='resnet50',
        help='model name:[resnet50, hrnetv2_32, PoolFormer_S24]')
    parser.add_argument("--model", type=str, default='PANet',
        help='model name:[PANet]')

    parser.add_argument("--epochs", type=int, default=80,
                        help="epoch number (default: 60)")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="total_itrs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
  
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=256)

    parser.add_argument("--n_cpu", type=int, default=8,
                        help="download datasets")
    
    parser.add_argument("--ckpt", type=str,
            default=None, help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='bi', 
                        help="loss type:[bce, bi]")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 100)")

    return parser


def get_dataset(opts):

    train_dst = EORSSDDataset(is_train=True,voc_dir=opts.trainset_path, trainsize=opts.trainsize)
    val_dst = EORSSDDataset(is_train=False,voc_dir=opts.testset_path, trainsize=opts.trainsize)
    return train_dst, val_dst


def main():
    if not os.path.exists('CHKP'):
        utils.mkdir('CHKP')

    opts = get_argparser().parse_args()

    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    opts.total_itrs=opts.epochs * (len(train_dst) // opts.batch_size)
    print('opts:',opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
        drop_last=True)  
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))


    model = eval(opts.model)(n_channels=3, backbone_name=opts.backbone)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    metrics=SODMetrics(cuda=True)

    opts.model+='_'+opts.backbone
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    if opts.loss_type=='bce':
        creterion=bce_DS4
    elif opts.loss_type=='bi':
        creterion=bi_DS4
    

    def save_ckpt(path):
        torch.save({
            "epoch": epoch+1,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)  
        
    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model=model.to(device)
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]   
        
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model=model.to(device)


    for epoch in range(cur_epoch,opts.epochs):
        model.train()
        cur_itrs=0
        data_loader = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        
        for (images, labels) in data_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
           
            S= model(images)
            total_loss=creterion(S[0], S[1], S[2], S[3], labels)
            
            running_loss += total_loss.data.item()

            total_loss.backward()
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, loss={:.4f}".format(epoch, opts.epochs, running_loss/cur_itrs)
            
            scheduler.step()

        if (epoch+1) % opts.val_interval == 0:
            save_ckpt('CHKP/latest_{}_{}_{}.pth'.format(opts.model, 
                                        opts.dataset, opts.loss_type))
        


if __name__ == '__main__':
    main()
