import argparse
from pickletools import optimize
from sched import scheduler
import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from classifier_train import Train
from data import Data_mura


parser = argparse.ArgumentParser(description='PyTorch MURA shoulder fracture classification')

parser.add_argument("--workers", default=2, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--batch_size", default=48, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--seed", default=4098, type=int)


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda')

    tfs_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    tfs_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_ds = Data_mura(tfs_train, 'train')
    val_ds = Data_mura(tfs_val, 'val')

    train_loader = DataLoader(
        train_ds,
        batch_size=48,
        num_workers=2,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=48,
        num_workers=2,
        shuffle=True,
        pin_memory=True
    )

    model = models.resnet18(weights=None, num_classes=2)
    optimizer = optim.Adam(model.parameters(), args.lr)
    train_process = Train(model=model, optimizer=optimizer, args=args)
    train_process.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
