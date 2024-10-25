from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data

from network.PANet import PANet
import torch.nn.functional as F
import torch
import torch.nn as nn
from datasets.AB2019BAS_Testset import AB2019BASDataset

from PIL import Image
import collections
import imageio



def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--testset_path", type=str, 
        default='../Dataset/AB2019BAS/Test',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='AB2019BAS', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')

    parser.add_argument("--model", type=str, default='PANet',
        help='model name:[PANet]')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=512)

    parser.add_argument("--n_cpu", type=int, default=1,
                        help="download datasets")
    
    parser.add_argument("--ckpt", type=str,
            default='',
              help="restore from checkpoint")
   
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--pred_path", type=str, default='../AB2019_Preds_2024/',
                        help="random seed (default: 1)")

    return parser


def get_dataset(opts):
    val_dst = AB2019BASDataset(is_train=False,voc_dir=opts.testset_path)
    return val_dst

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    

    opts = get_argparser().parse_args()
    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    opts.pred_path+=opts.model+'/'
    if not os.path.exists(opts.pred_path):
        os.makedirs(opts.pred_path, exist_ok=True)

    test_dst = get_dataset(opts)
    
    print('opts:',opts)
  
    test_loader = data.DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Test set: %d" %
          (opts.dataset, len(test_dst)))

    model = eval(opts.model)()
        
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(checkpoint)
            print('try: load pth from:', opts.ckpt)
        except:
            dic = collections.OrderedDict()
            for k, v in checkpoint["model_state"].items():
                #print( k)
                mlen=len('module')+1
                newk=k[mlen:]
                # print(newk)
                dic[newk]=v
            model.load_state_dict(dic)
            print('except: load pth from:', opts.ckpt)
    
    else:
        print("[!] Retrain")
     
     
    model.to(device)
    model.eval()
    data_loader = tqdm(test_loader, file=sys.stdout)

    
    for batch in data_loader:
        imgs,  name=batch['img'], batch['name']

        imgs = imgs.to(device, dtype=torch.float32)
       
        
        s1,s2,s3,s4, s1_sig,s2_sig,s3_sig,s4_sig = model(imgs)
        res=s1
      
        # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(opts.pred_path+name[0]+'.png', res)
        
    
    

if __name__ == '__main__':
    main()
