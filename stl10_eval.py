import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision

from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch MURA shoulder fracture classification')
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--seed", default=4098, type=int)
parser.add_argument("--model", default="simclr", type=str)
parser.add_argument('--root', default="/mnt/hdd/medical-imaging/data/", type=str, help='root')
parser.add_argument('--data', default="stl10", type=str, help='dataset')


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


args = parser.parse_args()
args.device = torch.device('cuda')
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


# model = ResNetSimCLR(base_model="resnet50", out_dim=128)
model = torchvision.models.resnet18(weights=None, num_classes=10).to(args.device)

args.checkpoint_path = "/mnt/hdd/medical-imaging/models/pre_train_2023_aug/Mar05_03-30-56/checkpoint_0200.pth.tar"
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


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(args.device)

writer = SummaryWriter(log_dir="/mnt/hdd/medical-imaging/models/classifier_stl10_aug/{}".format(datetime.now().strftime("%b%d_%H-%M-%S")))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

epochs = 300
for epoch in range(epochs):
    top1_train_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(val_loader):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        logits = model(x_batch)
    
        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]
    
    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    writer.add_scalar("train top1 acc", top1_train_accuracy, epoch)
    writer.add_scalar("val top1 acc", top1_accuracy, epoch)

