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
from dist_util import setup, cleanup, run_demo
from torch.nn.parallel import DistributedDataParallel as DDP

def train(encoder_decoder, predictor, rank, train_loader, optimizer, epoch, loss_func):
    encoder_decoder.train()
    predictor.eval()
    cnt = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched["heatmap"]
        target_reuse_dis = sample_batched["reuse_dis"]
        real_heatmap = data.to(rank)
        target_reuse_dis = target_reuse_dis.to(rank)
        optimizer.zero_grad()
        predictor.zero_grad()

        proxy_heatmap = encoder_decoder(real_heatmap)

        # get predictor miss rate of proxy heatmap
        proxy_reuse_dis = predictor(proxy_heatmap)
        loss = loss_func(proxy_reuse_dis, target_reuse_dis)
        # not only predicted miss rate, but proxy benchmark should also has close memory reference
        # real # of memory reference
        real_memory_reference = real_heatmap.sum()
        proxy_memory_reference = proxy_heatmap.sum()
        loss += (torch.abs(real_memory_reference-proxy_memory_reference)/real_memory_reference)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(encoder_decoder.parameters(), max_norm=2.0, norm_type=2)

        optimizer.step()

        if rank == 0:
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    predict_real_reuse_dis = predictor(real_heatmap)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Real Error rate: {:.6f} Predict Error rate: {:.6f} ".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        torch.mean(torch.abs((proxy_reuse_dis-target_reuse_dis)/(target_reuse_dis+1))),
                        torch.mean(torch.abs((predict_real_reuse_dis-target_reuse_dis)/(target_reuse_dis+1))),
                    )
                )


def main(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(42)

    train_kwargs = {"batch_size": 64}

    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)

    train_heatmap_dataset = HeatmapDataset(
        reuse_dis_file="../train_data/label.log",
        heatmap_dir="../train_data/npy_heatmap",
        transform=transforms.Compose([ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(train_heatmap_dataset, **train_kwargs)

    encoder_decoder = AutoEncoder(n_channels=1).to(rank).float()
    ddp_encoder_decoder = DDP(encoder_decoder, device_ids=[rank])
    reuse_dis_predictor = ReuseDisPredictor().to(rank).float()
    ddp_reuse_dis_predictor = DDP(reuse_dis_predictor, device_ids=[rank])
    ddp_reuse_dis_predictor.load_state_dict(torch.load('./ckpt/predictor_epoch_9.pth'))

    loss_func = torch.nn.MSELoss().to(rank)
    optimizer = torch.optim.Adam(ddp_encoder_decoder.parameters(), lr=1,
                             weight_decay=1e-5)

    for epoch in range(1, 11):
        train(ddp_encoder_decoder, ddp_reuse_dis_predictor, rank, train_loader, optimizer, epoch, loss_func)
    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(main, world_size)