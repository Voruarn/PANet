from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data
from datasets.LC8BASDataset import LC8BASDataset
from utils import ext_transforms as et
from network.PANet import PANet
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import pytorch_iou


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainset_path", type=str, 
        default='../Dataset/HRBAS/Train',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='../Dataset/HRBAS/Test',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='HRBAS', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
  
    parser.add_argument("--model", type=str, default='PANet',
        help='model name:[PANet]')
    parser.add_argument("--threshold", type=float, default=0.5,
                    help='threshold to predict foreground')
    parser.add_argument("--epochs", type=int, default=80,
                        help="epoch number (default: 60)")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="total_itrs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
  
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=1024)

    parser.add_argument("--n_cpu", type=int, default=4,
                        help="download datasets")
    
    parser.add_argument("--ckpt", type=str,
            default=None, help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='bi', 
                        help="loss type:[bce, bi, bsi]")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--save_path", type=str, default='./CHKP_HRBAS/',
                        help="epoch interval for eval (default: 100)")
    
    return parser


def get_dataset(opts):
    train_transform = et.ExtCompose([
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    train_dst = LC8BASDataset(is_train=True,voc_dir=opts.trainset_path, 
                                transform=train_transform)
    val_dst = LC8BASDataset(is_train=False,voc_dir=opts.testset_path,
                            transform=val_transform)
    return train_dst, val_dst


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
IOU = pytorch_iou.IOU(size_average = True)


def main():

    opts = get_argparser().parse_args()
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    tb_writer = SummaryWriter()
    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
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
    # val_loader = data.DataLoader(
    #     val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))


    model = eval(opts.model)()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=opts.weight_decay)
 
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)


    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])

        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")

    model=model.to(device)

    for epoch in range(cur_epoch,opts.epochs):
        model.train()
        cur_itrs=0
        data_loader = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        
        for (images, gts) in data_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            gts = gts.to(device, dtype=torch.float32)
            gts=gts.unsqueeze(dim=1)
            optimizer.zero_grad()
           
            s1,s2,s3,s4, s1_sig,s2_sig,s3_sig,s4_sig= model(images)
            
            loss1 = CE(s1, gts) + IOU(s1_sig, gts)
            loss2 = CE(s2, gts) + IOU(s2_sig, gts)
            loss3 = CE(s3, gts) + IOU(s3_sig, gts)
            loss4 = CE(s4, gts) + IOU(s4_sig, gts)
    
            total_loss = loss1 + loss2/2 + loss3/4 +loss4/8 
            
            running_loss += total_loss.data.item()

            total_loss.backward()
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, loss={:.4f}".format(epoch, opts.epochs, running_loss/cur_itrs)
            
            scheduler.step()
            

        if (epoch+1) % opts.val_interval == 0:
            torch.save(model.state_dict(), opts.save_path+'latest_{}_{}.pth'.format(opts.model, opts.dataset))


        tags = ["train_loss", "learning_rate"]

        tb_writer.add_scalar(tags[0], (running_loss/cur_itrs), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)



if __name__ == '__main__':
    main()
