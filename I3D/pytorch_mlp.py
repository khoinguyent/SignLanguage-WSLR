import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys

class MLP(nn.Module):
    def __init__(self, num_classes, total_frames):
      super().__init__()
      self.num_cls = num_classes
      self.layer1 = nn.Linear(num_classes * 2, 512)
      self.layer2 = nn.ReLU()
      self.layer4 = nn.Dropout(0.5)
      self.layer5 = nn.Linear(512, self.num_cls)

    def forward(self, x):
      output = self.layer1(x)
      output = self.layer2(output)
      output = self.layer4(output)
      output = self.layer5(output)
      return output