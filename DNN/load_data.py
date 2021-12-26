from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, miss_rate = sample["heatmap"], sample["miss_rate"]
        return {
            "heatmap": torch.unsqueeze(torch.from_numpy(image), dim=0).float(),
            "miss_rate": torch.unsqueeze(torch.as_tensor(miss_rate, dtype=torch.float32), dim=0),
        }


def read_miss_rate(miss_rate_file):
    def is_benchmark(line_string):
        if line_string.find("GemsFDTD") != -1:
            return "GemsFDTD"
        if line_string.find("bwaves") != -1:
            return "bwaves"
        if line_string.find("bzip2") != -1:
            return "bzip2"
        if line_string.find("cactusADM") != -1:
            return "cactusADM"
        if line_string.find("mcf") != -1:
            return "mcf"
        if line_string.find("zeusmp") != -1:
            return "zeusmp"
        return None

    miss_rate_list = []
    label_list = []
    with open(miss_rate_file, "r") as f:
        curr_benchmark = None
        for line in f.readlines():
            if is_benchmark(line.strip()):
                curr_benchmark = line.strip()
            if line.find("interval") != -1:
                # get interval
                interval_start = line.split()[1].strip()
                interval_end = line.split()[2].strip()
                miss_rate_list.append(float(line.split()[-1]))
                label_list.append("{}_inst_{}_{}.png".format(curr_benchmark, interval_start, interval_end))
    return miss_rate_list, label_list


class HeatmapDataset(Dataset):
    def __init__(self, miss_rate_file, heatmap_dir, transform=None):
        """
        Args:
            miss_rate_file (string): Path to the miss rate log.
            heatmap_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.miss_rate_list, self.label_list = read_miss_rate(miss_rate_file)
        self.heatmap_dir = heatmap_dir
        self.transform = transform

    def __len__(self):
        return len(self.miss_rate_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.heatmap_dir, self.label_list[idx])
        image = io.imread(img_name)
        sample = {"heatmap": image, "miss_rate": self.miss_rate_list[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    heatmap_dataset = HeatmapDataset(miss_rate_file="../heatmap/miss_rate.log", heatmap_dir="../heatmap/dataset")

    fig = plt.figure()

    for i in range(len(heatmap_dataset)):
        sample = heatmap_dataset[i]
        print(i, sample["heatmap"].shape, sample["miss_rate"])

        if i == 3:
            break
