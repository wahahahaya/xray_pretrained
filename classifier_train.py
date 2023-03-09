import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import save_config_file, save_checkpoint
from torchmetrics.functional import auroc
from torchmetrics.functional import cohen_kappa
from datetime import datetime


class Train(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']

        self.writer = SummaryWriter(log_dir="/mnt/hdd/medical-imaging/models/classifier_stl10_aug/{}".format(datetime.now().strftime("%b%d_%H-%M-%S")))
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, "training.log"), level=logging.INFO)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def accuracy(self, output, target, topk=(1,)):
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

    def train(self, train_loader, val_loader):
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start Classification training for {self.args.epochs} epochs.")
        logging.info(f"Seed: {self.args.seed}.")
        logging.info(f"Model: {self.args.model}.")
        logging.info(f"data: {self.args.data}.")
        if self.args.model == "simclr":
            logging.info(f"Check point: {self.args.checkpoint_path}.")

        for epoch in range(self.args.epochs):
            top1_train_accuracy = 0
            for counter, (image, label) in enumerate(train_loader):
                images = image.to(self.args.device)
                labels = label.to(self.args.device)

                out = self.model(images)
                loss = self.criterion(out, labels)

                top1 = self.accuracy(out, labels, topk=(1,))
                top1_train_accuracy += top1[0]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            top1_train_accuracy /= (counter + 1)
            top1_accuracy = 0
            top5_accuracy = 0
            for counter, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                logits = self.model(x_batch)
            
                top1, top5 = self.accuracy(logits, y_batch, topk=(1,5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]
            
            top1_accuracy /= (counter + 1)
            top5_accuracy /= (counter + 1)
            print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
            self.writer.add_scalar("train top1 acc", top1_train_accuracy, epoch)
            self.writer.add_scalar("val top1 acc", top1_accuracy, epoch)        

        logging.info("Training has finished.")
        # save model
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

