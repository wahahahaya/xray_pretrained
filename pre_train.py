import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.resnet_simclr import ResNetSimCLR
from pre_train_simclr_v2 import SimCLR

from torch.profiler import profile, record_function, ProfilerActivity

from torchvision import transforms, datasets
from data import Data_chest, Data_mura_simclr

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--loss', default="CE", type=str, help='SimCLR loss function.')
parser.add_argument('--data', default="chest", type=str, help='dataset')
parser.add_argument('--root', default="/mnt/hdd/medical-imaging/data/", type=str, help='root')


class GaussianNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            img = img + self.sigma*torch.rand_like(img)
            img = img.squeeze()
        img = self.tensor_to_pil(img)

        return img


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1


    # ColorJitter: brightness, contrast, saturation, hue
    # color_jitter = transforms.ColorJitter(0.2, 0.2, 0.0, 0.0)
    # tfs = transforms.Compose([
    #     transforms.RandomRotation(20),
    #     transforms.RandomCrop((224,224)),
    #     transforms.RandomApply([color_jitter], p=0.8),
    #     GaussianNoise(0.1),
    #     transforms.GaussianBlur(5, (0.1, 2.0)),
    #     transforms.ToTensor(),
    # ])

    tfs = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop((224,224), scale=(0.5,1.0), ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
        transforms.ToTensor(),
    ])



    if args.data == "chest":
        train_dataset = Data_chest(transforms_=tfs, root=args.root, mode='train')
        val_dataset = Data_chest(transforms_=tfs, root=args.root, mode='val')
    elif args.data == "MURA":
        train_dataset = Data_mura_simclr(transforms_=tfs, root=args.root, mode='train')
        val_dataset = Data_mura_simclr(transforms_=tfs, root=args.root, mode='val')
    elif args.data == "stl10":
        train_dataset = datasets.STL10(root=args.root, split='train+unlabeled', transform=ContrastiveLearningViewGenerator(tfs, 2), download=True)
        val_dataset = datasets.STL10(root=args.root, split='test', transform=ContrastiveLearningViewGenerator(tfs, 2), download=True)



    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer_SGD = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    # It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
