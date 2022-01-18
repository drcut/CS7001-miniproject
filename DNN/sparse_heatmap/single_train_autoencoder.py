import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from load_data import ToTensor, HeatmapDataset
from reuse_predictor import ReuseDisPredictor
from autoencoder import AutoEncoder

width = 4096
height = 1024

def train(encoder_decoder, train_loader, optimizer, epoch, loss_func):
    encoder_decoder.decoder.train()
    encoder_decoder.encoder.train()
    cnt = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched["heatmap"].cuda()
        addr_list = sample_batched["addr_list"].cuda()
        real_heatmap = data.view(-1, height, width, 1)
        optimizer.zero_grad()
        x_sp = encoder_decoder.encode(real_heatmap)
        proxy_heatmap = encoder_decoder.decode(x_sp)
        # (N, W)->(N*W)
        addr_list = addr_list.view(-1)
        # (N, C, H, W)->(N*C*W,H)
        tmp = proxy_heatmap.permute(0,1,3,2).reshape(-1,height)
        output = F.softmax(tmp, dim=1)
        loss = loss_func(output, addr_list)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(encoder_decoder.decoder.parameters(), max_norm=2.0, norm_type=2)

        # for param in encoder_decoder.decoder.parameters():
        #     if param.requires_grad:
        #         print(param.data)
        #         print(param.grad)
        optimizer.step()

        if True:
            if batch_idx % 10 == 0:
                batch_size = len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(addr_list.view_as(pred)).sum().item()/pred.shape[0]
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Correct rate: {:.6f} ".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        correct,
                    )
                )


def main():
    torch.manual_seed(42)

    train_kwargs = {"batch_size": 16}

    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)

    train_heatmap_dataset = HeatmapDataset(
        reuse_dis_file="/data/robinhan/proxy_benchmark/train_data/4096_label.log",
        trace_dir="/data/robinhan/proxy_benchmark/train_data/4096_trace",
        transform=transforms.Compose([ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(train_heatmap_dataset, **train_kwargs)

    encoder_decoder = AutoEncoder().float().cuda()
    encoder_decoder.encoder.load_state_dict(torch.load('./ckpt/predictor_epoch_14.pth'),strict=False)

    loss_func = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(encoder_decoder.decoder.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(encoder_decoder.decoder.parameters(), lr=0.01)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 11):
        train(encoder_decoder, train_loader, optimizer, epoch, loss_func)
        # torch.save(encoder_decoder.state_dict(), './ckpt/AE_epoch_{}.pth'.format(epoch))
        # scheduler.step()

if __name__ == "__main__":
    main()