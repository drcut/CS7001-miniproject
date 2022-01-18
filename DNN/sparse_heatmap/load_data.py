from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

height = 1024
width = 4096
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, reuse_dis, addr_list = sample["heatmap"], sample["reuse_dis"], sample["addr_list"]
        return {
            "heatmap": torch.unsqueeze(image, dim=0).float(),
            "reuse_dis": torch.unsqueeze(torch.as_tensor(reuse_dis, dtype=torch.float32), dim=0),
            "addr_list": torch.as_tensor(addr_list, dtype=torch.long),
        }


def read_reuse_dis(reuse_dis_file):
    reuse_dis_list = []
    label_list = []
    curr_tuple = []
    curr_cache_line_tuple = []
    with open(reuse_dis_file, "r") as f:
        curr_benchmark = None
        for line in f.readlines():
            if line.strip().find('log')>0:
                label_list.append(line.strip())

                if len(curr_cache_line_tuple)>0:
                    curr_tuple.append(curr_cache_line_tuple)
                curr_cache_line_tuple = []
                if len(curr_tuple)>0:
                    reuse_dis_list.append(curr_tuple)
                curr_tuple = []
            elif line.find('line')>0:
                if len(curr_cache_line_tuple)>0:
                    curr_tuple.append(curr_cache_line_tuple)
                curr_cache_line_tuple = []
            else:
                curr_cache_line_tuple.append(int(line.split()[-1]))
    return reuse_dis_list, label_list


class HeatmapDataset(Dataset):
    def __init__(self, reuse_dis_file, trace_dir, transform=None):
        self.reuse_dis_list, self.label_list = read_reuse_dis(reuse_dis_file)
        self.trace_dir = trace_dir
        self.transform = transform

    def __len__(self):
        return len(self.reuse_dis_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace_path = os.path.join(self.trace_dir, self.label_list[idx])
        # construct dense tensor
        heatmap = torch.zeros([height, width])
        cnt = 0
        addr_list = []
        with open(trace_path,'r') as f:
            for trace in f.readlines():
                addr = (int(int(trace.strip(),16)/16))%height
                heatmap[addr][cnt] = 1
                addr_list.append(addr)
                cnt+=1
        sample = {"heatmap": heatmap, "reuse_dis": self.reuse_dis_list[idx],"addr_list": addr_list}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    heatmap_dataset = HeatmapDataset(reuse_dis_file="/data/robinhan/proxy_benchmark/train_data/4096_label.log", \
                                    trace_dir="/data/robinhan/proxy_benchmark/train_data/4096_trace",\
                                    transform=transforms.Compose([ToTensor()]))

    fig = plt.figure()

    for i in range(len(heatmap_dataset)):
        sample = heatmap_dataset[i]
        print(i, sample["reuse_dis"].shape)
        if i == 5:
            break