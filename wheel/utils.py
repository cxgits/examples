import argparse
import shutil

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import random
import os


def func_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--cpdir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ga_batches", type=int, default=1)
    parser.add_argument("--val_epochs", type=int, default=1)
    parser.add_argument("--save_epochs", type=int, default=1)
    opt = parser.parse_args()
    return opt


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def func_folder(opt):
    os.makedirs(opt.cpdir, exist_ok=True)
    shutil.rmtree(opt.cpdir)
    os.makedirs(opt.cpdir)

    os.makedirs(opt.logdir, exist_ok=True)
    shutil.rmtree(opt.logdir)
    os.makedirs(opt.logdir)


def func_loss(x, y, logits):
    loss = {"loss_0": nn.CrossEntropyLoss()(logits, y)}
    return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
