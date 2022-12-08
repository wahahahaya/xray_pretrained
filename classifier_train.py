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

        self.writer = SummaryWriter(log_dir="classifier/{}".format(datetime.now().strftime("%b%d_%H-%M-%S")))
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, "training.log"), level=logging.INFO)
        self.criterion = torch.nn.NLLLoss().to(self.args.device)
        self.softmax = torch.nn.Softmax(dim=1)

    def train(self, train_loader, val_loader):
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start Classification training for {self.args.epochs} epochs.")
        logging.info(f"Seed: {self.args.seed}.")
        logging.info(f"Model: {self.args.model}.")
        if self.args.model == "simclr":
            logging.info(f"Check point: {self.args.checkpoint_path}.")

        for epoch in range(self.args.epochs):
            for image, label in tqdm(train_loader):
                images = image.to(self.args.device)
                labels = label.to(self.args.device)

                out = self.model(images)
                pred = self.softmax(out)
                loss = self.criterion(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.writer.add_scalar("cls loss", loss, epoch)

            pred_all = []
            label_all = []
            pre_correct = 0
            for image, label in tqdm(val_loader):
                images = image.to(self.args.device)
                labels = label.to(self.args.device)

                with torch.no_grad():
                    out = self.model(images)
                    pred = self.softmax(out)

                pre_correct += (pred.argmax(-1) == labels).float().sum()
                pred_all.extend(pred.cpu().detach().numpy())
                label_all.extend(labels.cpu().detach().numpy())


            label_all = torch.tensor(np.array(label_all))
            pred_all = torch.tensor(np.array(pred_all))
            auroc_score = auroc(pred_all, label_all, num_classes=2)
            kappa_score = cohen_kappa(pred_all, label_all, num_classes=2)
            top1 = pre_correct / len(label_all)
            self.writer.add_scalar("cls AUROC", auroc_score, epoch)
            self.writer.add_scalar("cls KAPPA", kappa_score, epoch)
            self.writer.add_scalar("cls top1 acc", top1, epoch)

            print("Epoch: %d, top1: %.4f, AUROC: %.4f, KAPPA: %.4f" % (epoch, top1, auroc_score, kappa_score))
            logging.info(f"Training Epoch: {epoch}\tLoss: {loss.item()}\tTop1 accuracy: {top1.item()}\tAUROC: {auroc_score}")

        logging.info("Training has finished.")
        # save model
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

