from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from load_data import ToTensor, HeatmapDataset
from miss_rate_predictor import MissRatePredictor

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class autoencoder(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(autoencoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)

        self.up1 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.up1(x)
        logits = self.outc(x)
        return logits

def train(args, encoder_decoder, predictor, device, train_loader, optimizer, epoch):
    encoder_decoder.train()
    predictor.eval()
    cnt = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched["heatmap"]
        target_miss_rate = sample_batched["miss_rate"]
        real_heatmap = data.to(device)
        target_miss_rate = target_miss_rate.to(device)
        optimizer.zero_grad()

        proxy_heatmap = encoder_decoder(real_heatmap)

        # get predictor miss rate of proxy heatmap
        proxy_predict_miss_rate = predictor(proxy_heatmap)
        loss = torch.nn.MSELoss()(proxy_predict_miss_rate, target_miss_rate)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(encoder_decoder.parameters(), max_norm=2.0, norm_type=2)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            predict_miss_rate = predictor(real_heatmap)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Real Error rate: {:.6f} Predict Error rate: {:.6f} ".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    torch.mean(torch.abs((proxy_predict_miss_rate-target_miss_rate)/target_miss_rate)),
                    torch.mean(torch.abs((predict_miss_rate-target_miss_rate)/target_miss_rate)),
                )
            )
            if args.dry_run:
                break


def test(encoder_decoder, predictor, device, test_loader):
    encoder_decoder.eval()
    predictor.eval()
    proxy_error_rate = 0
    real_error_rate = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            cnt+=1
            data = sample_batched["heatmap"]
            target_miss_rate = sample_batched["miss_rate"]
            real_heatmap = data.to(device)
            target_miss_rate = target_miss_rate.to(device)
            proxy_heatmap = encoder_decoder(real_heatmap)

            # get predictor miss rate of proxy heatmap
            proxy_predict_miss_rate = predictor(proxy_heatmap)
            predict_miss_rate = predictor(real_heatmap)

            proxy_error_rate += torch.mean(torch.abs((proxy_predict_miss_rate-target_miss_rate)/target_miss_rate))
            real_error_rate += torch.mean(torch.abs((predict_miss_rate-target_miss_rate)/target_miss_rate))

    proxy_error_rate /= cnt
    real_error_rate /= cnt

    print("Test set: Proxy LLC Error-rate: {:.4f}  Real LLC EDrror-rate: {:.4f}".format(proxy_error_rate, real_error_rate))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Generate Proxy benchmark")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    # device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_heatmap_dataset = HeatmapDataset(
        miss_rate_file="../heatmap/dataset/train/miss_rate.log",
        heatmap_dir="../heatmap/dataset/train",
        transform=transforms.Compose([ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(train_heatmap_dataset, **train_kwargs)

    valid_heatmap_dataset = HeatmapDataset(
        miss_rate_file="../heatmap/dataset/valid/miss_rate.log",
        heatmap_dir="../heatmap/dataset/valid",
        transform=transforms.Compose([ToTensor()]),
    )
    valid_loader = torch.utils.data.DataLoader(valid_heatmap_dataset, **test_kwargs)

    encoder_decoder = autoencoder(n_channels=1).to(device).float()
    miss_rate_predictor = MissRatePredictor().to(device).float()
    miss_rate_predictor.load_state_dict(torch.load('llc_predictor.pth'))

    optimizer = torch.optim.Adam(encoder_decoder.parameters(), lr=args.lr,
                             weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        train(args, encoder_decoder, miss_rate_predictor, device, train_loader, optimizer, epoch)
        test(encoder_decoder, miss_rate_predictor, device, valid_loader)
    # torch.save(encoder_decoder.state_dict(), './autoencoder.pth')


if __name__ == "__main__":
    main()
