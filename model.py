import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import config

class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        mask = (diff < self.omega).float()
        
        loss = mask * self.omega * torch.log(1 + diff / self.epsilon) + \
               (1 - mask) * (diff - self.C)
        
        return loss.mean()

class FaceAlignmentModel(pl.LightningModule):
    def __init__(self, model_type="resnet", loss_type="mse"):
        super(FaceAlignmentModel, self).__init__()
        self.save_hyperparameters()
        self.loss_type = loss_type
        
        if model_type == "resnet":
            self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=config.NUM_POINTS * 2)
        elif model_type == "efficientnet":
            self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=config.NUM_POINTS * 2)
        else:
            raise ValueError("Unsupported model type")
            
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "wing":
            self.criterion = WingLoss()
        else:
            raise ValueError("Unsupported loss type")
        
    def forward(self, x):
        out = self.backbone(x)
        out = out.view(-1, config.NUM_POINTS, 2)
        return out
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        landmarks = batch["landmarks"]
        preds = self.forward(images)
        loss = self.criterion(preds, landmarks)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        landmarks = batch["landmarks"]
        preds = self.forward(images)
        loss = self.criterion(preds, landmarks)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        return [optimizer], [scheduler]
