from __future__ import print_function
import argparse
import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import contextlib
import torch.cuda.amp
import random


@contextlib.contextmanager
def identity_ctx():
    yield

cache_line_size_cnt = 5
reuse_dis_hist = 6
width = 4096
height = 1024
batch = 72

class ReuseDisPredictor(nn.Module):
    def __init__(self):
        super(ReuseDisPredictor, self).__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv2d(1, 64, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            spconv.SparseMaxPool2d(2, 2),
            spconv.SubMConv2d(64, 64, 3, 1),
            nn.BatchNorm1d(64),
            spconv.SparseMaxPool2d(2, 2),
            spconv.SubMConv2d(64, 64, 3, 1),
            nn.BatchNorm1d(64),
            spconv.SparseMaxPool2d(2, 2),
            spconv.SubMConv2d(64, 64, 3, 1),
            nn.BatchNorm1d(64),
            spconv.SparseMaxPool2d(2, 2),
            spconv.SubMConv2d(64, 64, 3, 1),
            nn.BatchNorm1d(64),
            spconv.SparseMaxPool2d(2, 2),
            spconv.SubMConv2d(64, 64, 3, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseMaxPool2d(2, 2),
            spconv.ToDense(),
        )
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(65536, 1024)
        self.fc2 = nn.Linear(1024, cache_line_size_cnt*reuse_dis_hist)

    def forward(self, x: torch.Tensor):
        # x: must be NHWC tensor
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, height, width, 1))
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x = self.net(x_sp)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)
        output = output.view(-1, 1, cache_line_size_cnt,reuse_dis_hist)
        return output