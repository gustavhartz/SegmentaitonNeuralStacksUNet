import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
from utils.plotting import plot_images_grid

from utils.start_tensorboard import run_tensorboard
from data_utils import EM_DATA

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--unet_depth', type=int, default=2, help='Depth of UNet')
parser.add_argument('--n_gpus', type=int, default=0, help='number of GPUs')


opt = parser.parse_args()


##########################
######### MODEL ##########
##########################

def create_pred_image(x, y_hat, y):
    """
    Concatenate input, output and target for the purpose of illustration
    :param x:
    :param y_hat:
    :param y:
    :return: Grid created with pytorch
    """
    # predictions with input for illustration purposes
    grid = torchvision.utils.make_grid(torch.cat([x.cpu(), y.cpu(), y_hat.cpu()], dim=0), nrow=1)

    return grid


class SegmentationNeuralStacksUNet(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(SegmentationNeuralStacksUNet, self).__init__()

        # default config
        self.path = os.getcwd() + '/data'
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = opt.batch_size

    def forward(self, x):
        # x = x.to(device='cuda')

        output = self.model(x)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 1 == 0:
                final_image = create_pred_image(x, y_hat, y)
                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = EM_DATA(train=True, size=256, _transform=transform, data_path="Data")
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True)
        return train_loader

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        test_data = EM_DATA(train=False, size=256, _transform=transform, data_path="Data", validation=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=True)
        return test_loader


def run_trainer():
    conv_lstm_model = UNet(1, in_channels=1, depth=opt.unet_depth, merge_mode='concat')

    model = SegmentationNeuralStacksUNet(model=conv_lstm_model)

    trainer = Trainer(max_epochs=opt.epochs)

    trainer.fit(model)


if __name__ == '__main__':
    p1 = Process(target=run_trainer)  # start trainer
    p1.start()
    p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
    p2.start()
    p1.join()
    p2.join()
