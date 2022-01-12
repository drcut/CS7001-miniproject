from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from load_data import ToTensor, HeatmapDataset

cache_line_size_cnt = 6
reuse_dis_hist = 9
class MissRatePredictor(nn.Module):
    def __init__(self):
        super(MissRatePredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, 1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, 1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3, 1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, 3, 1)
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, cache_line_size_cnt*reuse_dis_hist)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = F.max_pool2d(x, 2)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.bn6(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)
        output = output.view(-1,1,cache_line_size_cnt,reuse_dis_hist)
        return output

loss_func = nn.MSELoss().cuda()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    cnt = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched["heatmap"]
        target_reuse_dis = sample_batched["reuse_dis"]
        data = data.to(device)
        target_reuse_dis = target_reuse_dis.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_func(output, target_reuse_dis)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Error rate: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    torch.mean(torch.abs((output-target_reuse_dis)/(target_reuse_dis+0.0001))),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    error_rate = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            cnt+=1
            data = sample_batched["heatmap"]
            target_reuse_dis = sample_batched["reuse_dis"]
            data = data.to(device)
            target_reuse_dis = target_reuse_dis.to(device)
            output = model(data)
            error_rate += torch.mean(torch.abs((output-target_reuse_dis)/target_reuse_dis)) 

    error_rate /= cnt

    print("Test set: LLC Error-rate: {:.4f}".format(error_rate))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_heatmap_dataset = HeatmapDataset(
        reuse_dis_file="train_data/pure_log_label/small_train_reuse_dis.log",
        heatmap_dir="train_data/npy_heatmap",
        transform=transforms.Compose([ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(train_heatmap_dataset, **train_kwargs)

    valid_heatmap_dataset = HeatmapDataset(
        reuse_dis_file="train_data/pure_log_label/small_valid_reuse_dis.log",
        heatmap_dir="train_data/npy_heatmap",
        transform=transforms.Compose([ToTensor()]),
    )
    valid_loader = torch.utils.data.DataLoader(valid_heatmap_dataset, **test_kwargs)

    model = MissRatePredictor().to(device).float()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, valid_loader)
        scheduler.step()
    torch.save(model.state_dict(), './llc_predictor.pth')


if __name__ == "__main__":
    main()
