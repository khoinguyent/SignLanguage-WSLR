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
from pytorchtools import EarlyStopping

import pandas as pd

import numpy as np

from config import Config
from pytorch_i3d import InceptionI3d

# from datasets.nslt_dataset import NSLT as Dataset
from datasets.nslt_dataset_multi import NSLT as Dataset
from datasets.asl_dataset import ASL as ASL_Dataset

from sklearn.model_selection import KFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#training with k-fold, using early stopping
def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None,
        dataset_name='WLASL',
        k_folds=5):
    
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                            videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = None
    dataloader = None
    val_dataset = None
    val_dataloader = None

    if(dataset_name == 'WLASL'):
    #RGB data stream
        dataset = torch.utils.data.ConcatDataset([Dataset(train_split, 'train', root, train_transforms),Dataset(train_split, 'test', root, test_transforms)])
        prefix = 'nslt_'

    elif(dataset_name == 'ASL'):
        dataset = torch.utils.data.ConcatDataset([ASL_Dataset(root, 'train', int(train_split), train_transforms),ASL_Dataset(root, 'test', int(train_split), test_transforms)])
        prefix = 'asl_'

    num_classes = dataset.num_classes

    #load model
    if(mode == 'flow'):
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    
    #replace logits
    i3d.replace_logits(num_classes)

    log_file = prefix + num_classes + "logs.csv"
    last_epoch = 0
    if(weights == None):
        if(os.path.exists(log_file)):
            os.remove(log_file)

        with open (log_file,'a') as logs:
            line = 'epoch\tacc_train\ttot_loss_train\tacc_val\ttotal_loss_train\n'
            logs.writelines(line)
    else:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

        #load the last epoch
        load_logs_data = pd.read_csv("logs.csv", sep='\t', engine='python')
        last_epoch = int(load_logs_data.tail(1).values.tolist()[0][0])

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    acc_train = 0.0
    tot_loss_train = 0.0

    acc_val = 0.0
    tot_loss_val = 0.0
    

    best_val_score = 0
    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

    kflod = Kfold(n_splits=k_folds,shuffle=True)