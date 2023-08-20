import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from model import ViT
import argparse


def func_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    args = parser.parse_args()
    return args


class BuildLightning(pl.LightningModule):
    def __init__(self, batch_size):
        super(BuildLightning, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.vit = ViT(image_size=28, channels=1, patch_size=4,
                       dim=32, depth=6, heads=16, mlp_dim=64, num_classes=10)
        self.celoss = nn.CrossEntropyLoss()
        self.accu = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.vit(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self.vit(x)
        loss = self.celoss(logit, y)
        self.log("loss", {"train_loss": loss}, reduce_fx="mean")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self.vit(x)
        loss = self.celoss(logit, y)
        self.log("loss", {"valid_loss": loss}, reduce_fx="mean")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit = self.vit(x)
        accu = self.func_metrics(logit, y, self.accu)
        return accu

    def test_epoch_end(self, outputs):
        print("----------------------------------------------------------")
        print("\n")
        print("test_mean_accu:", torch.Tensor(outputs).mean().float())
        print("\n")
        print("----------------------------------------------------------")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def func_metrics(self, logit, y, metrics):
        logit = nn.Softmax(dim=-1)(logit)
        pred = torch.argmax(logit, dim=-1)
        return metrics(pred, y)

    def prepare_data(self):
        datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())

    def setup(self, stage: str):
        train_data = datasets.MNIST("data", train=True, download=False, transform=transforms.ToTensor())
        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(train_data, [50000, 5000, 5000])

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        return train_dataloader

    def val_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return valid_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        return test_dataloader
