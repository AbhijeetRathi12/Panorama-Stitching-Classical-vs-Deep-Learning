
import torch.nn as nn
import sys
import torch
import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(delta, labels):
  
    criterion = nn.MSELoss()

    labels = labels.float()
    loss = torch.sqrt(criterion(delta, labels))
    return loss


class HomographyModel(pl.LightningModule):
    def _init_(self):
        super(HomographyModel, self)._init_()
        self.model = SuperNet()

    def forward(self, a):
        return self.model(a)

    def validation_step(self, batch, batch_idx):
        # img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(batch)
        loss = LossFn(delta, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out["val_loss"] for out in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

class SuperNet(nn.Module):
    def _init_(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super()._init_()
        # Spatial transformer localization-network
        self.conv1 = nn.Sequential(nn.Conv2d(2, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 8)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpool(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.maxpool(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out