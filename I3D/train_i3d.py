import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import transforms
import videotransforms
import pandas as pd

import numpy as np

from config import Config
from pytorch_i3d import InceptionI3d

# from datasets.nslt_dataset import NSLT as Dataset
from datasets.nslt_dataset import NSLT as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    #remove log files when train from begining
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow', default='rgb')
    parser.add_argument('-weights', type=str, defaullt=None)
    parser.add_argument('-save_model', type=str, default='checkpoints/')
    parser.add_argument('-root', type=str, default={'word': '../data/WLASL2000'})
    parser.add_argument('--num_class', type=int, default=2000)
    parser.add_argument('--config', type=int, default='configfiles/asl2000.ini')
    parser.add_argument('--train_split', type=int, default='preprocess/nslt_2000.json')

    args = parser.parse_args()

    mode = args.mode
    root = args.root
    weights = args.weights
    save_model = args.save_model
    num_class = args.num_class
    config_file = args.config
    train_split = args.train_split

    # WLASL setting
    # mode = 'rgb'
    # root = {'word': '../../data/WLASL2000'}

    # save_model = 'checkpoints/'
    # train_split = 'preprocess/nslt_2000.json'

    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    # weights = None
    # config_file = 'configfiles/asl2000.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
