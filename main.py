from __future__ import print_function
import argparse
import torch
import data_loader
from model.ADADDA import ADADDA
from trainer import Trainer
from utils.utils import *

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation for ADADDA')
    parser.add_argument('--backbone', type=str, default='resnet50_tripool', help='model type')
    parser.add_argument('--att_type', type=str, choices=['TripletAttention'],default='TripletAttention')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--l2_decay', type=float, default=5e-4)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    
    parser.add_argument('--root_path', type=str, default="/dat01/maxin2/data/transferlearning/")
    parser.add_argument('--source_dir', type=str, default="AID")
    parser.add_argument('--test_dir', type=str, default="NWPU")
    parser.add_argument('--gpu', type=str, default='0,1', help='GPU ID select')

    args = parser.parse_args()
    
    return args

def main(args):
    cuda_stat = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_stat else {}
    src_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    tar_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    tar_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
    num_gpu = set_gpu(args)

    model = ADADDA(args)
    if cuda_stat:
        model = torch.nn.DataParallel(model, list(range(num_gpu)))
        model.cuda()
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    my_trainer = Trainer(model, args.backbone, optimizer, lr_scheduler, cuda_stat, src_loader,\
                        tar_train_loader, tar_test_loader, args)
    my_trainer.fit()

if __name__ == '__main__':
    args = get_args()
    main(args)