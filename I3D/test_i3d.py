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
from datasets.nslt_dataset import NSLT_Test as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run(root, train_split, rgb_weights, flow_weights, configs):
    correct = 0
    correct_5 = 0
    correct_10 = 0

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'test', root, test_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                             pin_memory=True)

    num_classes = dataset.num_classes
    #load models
    i3d_flow = InceptionI3d(400, in_channels=2)
    i3d_flow.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    
    i3d_rgb = InceptionI3d(400, in_channels=3)
    i3d_rgb.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    i3d_flow.replace_logits(num_classes)
    i3d_rgb.replace_logits(num_classes)

    #load weights
    rgb_weights = torch.load(rgb_weights)
    flow_weights = torch.load(flow_weights)
    
    i3d_rgb.load_state_dict(rgb_weights, strict=True)
    i3d_flow.load_state_dict(flow_weights, strict=True)

    #cuda
    i3d_rgb.cuda()
    i3d_rgb = nn.DataParallel(i3d_rgb)

    i3d_flow.cuda()
    i3d_flow = nn.DataParallel(i3d_flow)

    i3d_rgb.train(False)
    i3d_flow.train(False)
    
    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    for data in dataloader:
        if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
            continue

        # inputs, labels, vid, src = data
        rgb_inputs, flow_inputs, labels, vid = data

        # wrap them in Variable
        rgb_inputs = rgb_inputs.cuda()
        t1 = rgb_inputs.size(2)

        flow_inputs = flow_inputs.cuda()
        t2 = flow_inputs.size(2)

        labels = labels.cuda()

        per_rgb_frame_logits = i3d_rgb(rgb_inputs, pretrained=False)
        # upsample to input size
        per_rgb_frame_logits = F.upsample(per_rgb_frame_logits, t1, mode='linear')

        per_flow_frame_logits = i3d_flow(flow_inputs, pretrained=False)
        # upsample to input size
        per_flow_frame_logits = F.upsample(per_flow_frame_logits, t2, mode='linear')

        per_frame_logits = (per_rgb_frame_logits + per_flow_frame_logits) / 2.0
        predictions = torch.mean(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy())

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(vid, float(correct) / len(dataloader), float(correct_5) / len(dataloader),
              float(correct_10) / len(dataloader))

    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_weights', type=str, default=None)
    parser.add_argument('--flow_weights', type=str, default=None)
    parser.add_argument('--root', type=str, default={'word': '../data/WLASL2000'})
    parser.add_argument('--config', type=str, default='configfiles/asl2000.ini')
    parser.add_argument('--train_split', type=str, default='preprocess/nslt_2000.json')

    args = parser.parse_args()

    root = args.root
    rgb_weights = args.rgb_weights
    flow_weights = args.flow_weights
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
    run(configs=configs, root=root, train_split=train_split, rgb_weights=rgb_weights, flow_weights=flow_weights)