import logging
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime

from utils import save_config_file, save_checkpoint, accuracy


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir="pre_train/{}".format(datetime.now().strftime("%b%d_%H-%M-%S")))
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'simclr.log'), level=logging.INFO)
        self.loss = self.args.loss
        if self.loss == "CE":
            self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        elif self.loss == "MSE":
            self.criterion = torch.nn.MSELoss().to(self.args.device)
        elif self.loss == "L1":
            self.criterion = torch.nn.L1Loss().to(self.args.device)

        self.criterion_factor = torch.nn.MSELoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        # normalize features, diagonal = 1
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        lables_onehot = F.one_hot(labels, num_classes=len(labels)-1).float()
        
        logits = logits / self.args.temperature
        return logits, labels, lables_onehot

    def sex_loss(self, factor, features):
        factor = factor.to(self.args.device)    # torch.Size([128])
        factor = factor.repeat(2)   # torch.Size([256])
        factor = (factor.unsqueeze(0) == factor.unsqueeze(1)).float()   # torch.Size([256, 256])

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        logits = similarity_matrix

        return logits, factor

    def age_loss(self, factor, features):
        factor = factor.to(self.args.device)
        factor = factor.repeat(2).float().unsqueeze(1)
        factor_map = torch.cdist(factor, factor, p=2)
        factor_map = torch.exp(-((factor_map**2)/200))

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        logits = similarity_matrix

        return logits, factor_map

    def train(self, train_loader, val_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.device}.")
        logging.info(f"model: {self.args.arch}.")
        logging.info(f"data: {self.args.data}.")
        logging.info(f"batch: {self.args.batch_size}.")
        logging.info(f"softmax temperature: {self.args.temperature}.")
        logging.info(f"info nce loss: {self.args.loss}.")
        logging.info(f"learning rate: {self.args.lr}.")

        for epoch_counter in range(self.args.epochs):
            for images, factor in tqdm(train_loader):
                images = torch.cat(images, dim=0).to(self.args.device)

                batch, channel, weidth, length = images.shape
                images = torch.broadcast_to(images, (batch, 3, weidth, length)).to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    # logits, labels = self.info_nce_loss(features)  # logits.shape == [256,255], labels.shape == [256]
                    logits, labels, lables_onehot = self.info_nce_loss(features)  # logits.shape == [256,255], labels.shape == [256]

                    # logits_sex, factor_sex = self.sex_loss(factor['sex'], features)
                    # logits_age, factor_age = self.age_loss(factor['age'], features)

                    # loss_sex = self.criterion_factor(logits_sex, factor_sex)
                    # loss_age = self.criterion_factor(logits_age, factor_age)
                    if self.loss == "CE":
                        loss_feature = self.criterion(logits, labels)
                    elif self.loss == "MSE":
                        loss_feature = self.criterion(logits, lables_onehot)

                    loss = loss_feature

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()


                top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.writer.add_scalar('pre loss', loss, global_step=epoch_counter)
            self.writer.add_scalar('pre loss feature', loss_feature, global_step=epoch_counter)
            self.writer.add_scalar('pre acc/top1', top1[0], global_step=epoch_counter)
            self.writer.add_scalar('pre acc/top5', top5[0], global_step=epoch_counter)


            for images_val, factor_val in tqdm(val_loader):
                images_val = torch.cat(images_val, dim=0).to(self.args.device)

                batch, channel, weidth, length = images_val.shape
                images_val = torch.broadcast_to(images_val, (batch, 3, weidth, length)).to(self.args.device)
                with torch.no_grad():
                    features_val = self.model(images_val)
                # logits, labels = self.info_nce_loss(features)  # logits.shape == [256,255], labels.shape == [256]
                logits_val, labels_val, lables_onehot_val = self.info_nce_loss(features_val)  # logits.shape == [256,255], labels.shape == [256]
                
                top1_val, top5_val = accuracy(logits_val, labels_val, topk=(1, 5))
            self.writer.add_scalar('pre val acc/top1', top1_val[0], global_step=epoch_counter)
            self.writer.add_scalar('pre val acc/top5', top5_val[0], global_step=epoch_counter)

        
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            if epoch_counter%100 == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.info(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop val accuracy: {top1_val[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
