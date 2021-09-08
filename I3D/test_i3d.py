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
import videoprocessing
import pandas as pd

import numpy as np

from config import Config
from pytorch_i3d import InceptionI3d
from pytorch_mlp import MLP


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run(video_path, num_classes, weights):
    video = video_path.split("/")[-1]
    video_path = video_path[0:len(video_path) - (len(video) + 2)]

    inputs_rgb = videoprocessing.video_to_tensor(
        videoprocessing.load_rgb_frames_from_video(video_path, video, 0, 64)
    )

    inputs_flow = videoprocessing.video_to_tensor(
        videoprocessing.load_flow_frames_upd(video_path, video, 0, 64)
    )

    #load models
    i3d_flow = InceptionI3d(400, in_channels=2)
    i3d_rgb = InceptionI3d(400, in_channels=3)
    mlp = MLP(num_classes, 64)

    #load weights
    weights = torch.load(weights)
    print(weights)
    i3d_rgb.load_state_dict(weights['rgb'], strict=False)
    i3d_flow.load_state_dict(weights['flow'], strict=False)
    mlp.load_state_dict(weights['mlp'], strict=False)

    #cuda
    i3d_rgb.cuda()
    i3d_rgb = nn.DataParallel(i3d_rgb)

    i3d_flow.cuda()
    i3d_flow = nn.DataParallel(i3d_flow)

    mlp.cuda()
    mlp = nn.DataParallel(mlp)

    #set model to test state
    i3d_rgb.train(False)
    i3d_flow.train(False)
    mlp.train(False)

    inputs_rgb.cuda()
    inputs_flow.cuda()

    i3d_rgb.eval()
    i3d_flow.eval()
    mlp.eval()

    t_rgb = inputs_rgb.size(2)
    t_flow = inputs_flow.szie(2)

    per_frame_logits_rgb = i3d_rgb(inputs_rgb)
    per_frame_logits_rgb = F.upsample(per_frame_logits_rgb, t_rgb, mode='linear')
    
    per_frame_logits_flow = i3d_flow(inputs_flow)
    per_frame_logits_flow = F.upsample(per_frame_logits_flow, t_flow, mode='linear')

    outputs = None
    #put output of rgb stream and flow stream through MLP network
    for i in range(0, per_frame_logits_flow.shape[2]):
        input_mlp = torch.cat((per_frame_logits_rgb[:,:,i], per_frame_logits_flow[:,:,i]), 1)
        output = mlp(input_mlp)

        if i == 0:
            outputs = output
        else:
            outputs = torch.cat((outputs, output))

    outputs = outputs.unsqueeze(0)
    outputs = torch.transpose(outputs, 1, 2)

    predictions = torch.max(outputs, dim=2)[0]


