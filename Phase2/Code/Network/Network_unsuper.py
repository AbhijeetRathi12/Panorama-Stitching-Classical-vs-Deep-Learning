
import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia

# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn(delta, labels):

    loss = F.l1_loss(delta, labels, reduction='mean')
    return loss

class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = UnsuperNet()

    def forward(self, I1, CoordinateBatch, Ca, Cb, Pa):
        return self.model(I1, CoordinateBatch, Ca, Cb, Pa)

    def training_step(self, batch, labels):
        # img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(batch)
        loss = LossFn(delta, labels)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, VI1, VCoordinateBatch, VCa, VCb, VPa, VPb): 
        # img_a, patch_a, patch_b, corners, gt = batch
        PbPredicted, H4ptPrecicted = self.model(VI1, VCoordinateBatch, VCa, VCb, VPa)
        loss = LossFn(PbPredicted, VPb)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out["val_loss"] for out in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

class UnsuperNet(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        
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

    def forward(self, Ibatch, CoordBatch, Ca, Cb, Pa):
        
        out = self.conv1(Ibatch)
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
        H4pt_predict = self.fc2(out)
        out = Tensor_DLT(H4pt_predict, Ca)
        out = torch.tensor(out)
        Pa = Pa.view(256,1,128,128)
        Pa = Pa.float()
        out = out.float()
        PB_pred = kornia.geometry.transform.warp_perspective(Pa, out, dsize = (128,128),
                                                            mode='bilinear', padding_mode='zeros', 
                                                            align_corners=True, fill_value=torch.zeros(3)).requires_grad_()

        return PB_pred, H4pt_predict

def Tensor_DLT(H4pt, C4pt_A):
    C4pt_B = H4pt + C4pt_A
    
    Aunderscore = torch.empty((8, 8), dtype=torch.float64)
    Hunderscore = torch.empty((8, 1), dtype=torch.float64)
    H_all = torch.empty((256, 3, 3), dtype=torch.float64)
    b_ = torch.empty((8, 1), dtype=torch.float64)

    values = [0, 2, 4, 6]
    for i in range(C4pt_A.shape[0]):
        for val in values:
            u_i = C4pt_A[i, val]
            v_i = C4pt_A[i, val + 1]
            u_pi = C4pt_B[i, val]
            v_pi = C4pt_B[i, val + 1]

            a = torch.tensor([0, 0, 0, -u_i, -v_i, -1, (v_pi * u_i), (v_i * v_pi)], dtype=torch.float64)
            b = torch.tensor([u_i, v_i, 1, 0, 0, 0, -u_pi * u_i, -u_pi * v_i], dtype=torch.float64)
            c = torch.tensor([-v_pi, u_pi], dtype=torch.float64)

            Aunderscore[val, :] = a
            Aunderscore[val + 1, :] = b
            b_[val:val + 2, 0] = c.t()

        A_inv = torch.pinverse(Aunderscore)
        h = torch.matmul(A_inv, b_)
        constant_term = torch.tensor([[1.]], dtype=torch.float64)
        Hunderscore = torch.cat((h, constant_term), dim=0)
        H_all[i, :, :] = Hunderscore.view(3, 3)

    H_all.requires_grad = True

    return H_all
