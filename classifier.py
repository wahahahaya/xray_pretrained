import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim

from classifier_train import Train
from data import Data_mura
from models.resnet_simclr import ResNetSimCLR


parser = argparse.ArgumentParser(description='PyTorch MURA shoulder fracture classification')

parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--weight_decay", default=8e-4, type=float)
parser.add_argument("--seed", default=4098, type=int)
parser.add_argument("--model", default="simclr", type=str)
parser.add_argument('--root', default="/mnt/hdd/medical-imaging/data/", type=str, help='root')
parser.add_argument('--data', default="stl10", type=str, help='dataset')


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda')
    # args.seed = random.randint(1, 10000)
    args.seed = 4562
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tfs_train = transforms.Compose([
        # transforms.Resize((96, 96)),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    tfs_val = transforms.Compose([
        # transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    if args.data == "mura":
        train_ds = Data_mura(transforms_=tfs_train, root=args.root, mode='train')
        val_ds = Data_mura(transforms_=tfs_val, root=args.root, mode='val')
    elif args.data == "stl10":
        train_ds = datasets.STL10(root=args.root, split='train', transform=tfs_train, download=True)
        val_ds = datasets.STL10(root=args.root, split='test', transform=tfs_val, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
        pin_memory=True
    )

    if args.model == "ImageNet":
        model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        dim_mlp = model.fc.in_features
        model.fc = nn.Linear(dim_mlp, 10)
    elif args.model == "scratch":
        model = models.resnet18(weights=None, num_classes=2)
    elif args.model == "simclr":
        model = models.resnet18(weights=None, num_classes=10).to(args.device)
        args.checkpoint_path = "/mnt/hdd/medical-imaging/models/pre_train_2023_aug/Mar08_07-34-00/checkpoint_0200.pth.tar"
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

        log = model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias


    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    train_process = Train(model=model, optimizer=optimizer, args=args)
    train_process.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
