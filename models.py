import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

import config
from models_utils import WingLoss, AdaptiveWingLoss, HeatmapFocalLoss, HeatmapHead

class FaceAlignmentModel(pl.LightningModule):
    def __init__(self, model_type: str = "efficientnet", head_type: str = "regression", loss_type: str = "mse"):
        super().__init__()

        self.num_points = config.NUM_POINTS
        self.img_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        self.img_h, self.img_w = int(self.img_size[0]), int(self.img_size[1])
        hm_def = (config.HEATMAP_SIZE, config.HEATMAP_SIZE)
        self.hm_h, self.hm_w = int(hm_def[0]), int(hm_def[1])
        self.lr = config.LEARNING_RATE

        self.head_type = head_type
        self.loss_type = loss_type
        
        if model_type == "efficientnet":
            self.backbone = timm.create_model("efficientnet_b0", pretrained=True, features_only=True, in_chans=3)
        elif model_type == "convnext":
            self.backbone = timm.create_model("convnextv2_nano", pretrained=True, features_only=True, in_chans=3)

        in_ch_last = self.backbone.feature_info.channels()[-1]

        if self.head_type == "regression":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch_last, in_ch_last // 2 if in_ch_last >= 64 else in_ch_last),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(in_ch_last // 2 if in_ch_last >= 64 else in_ch_last, self.num_points * 2)
            )
            if loss_type == "mse":
                self.criterion = nn.MSELoss(reduction='mean')
            elif loss_type == "wing":
                self.criterion = WingLoss()
            elif loss_type == "awing":
                self.criterion = AdaptiveWingLoss()

        elif self.head_type == "heatmap":
            self.heatmap_head = HeatmapHead(in_channels=in_ch_last, num_points=self.num_points, hm_h=self.hm_h, hm_w=self.hm_w)
            if loss_type == "mse":
                self.criterion = nn.MSELoss(reduction='mean')
            elif loss_type == "focal":
                self.criterion = HeatmapFocalLoss()
            elif loss_type == "bce":
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.HEATMAP_POS_WEIGHT))

    def forward(self, x):
        feats = self.backbone(x)[-1]
        if self.head_type == "regression":
            pooled = self.pool(feats)
            out = self.regressor(pooled)
            out = out.view(-1, self.num_points, 2)
        else:
            out = self.heatmap_head(feats)
        return out
    
    def _shared_step(self, batch, train=True):
        imgs = batch["image"]
        device = imgs.device
        preds = self.forward(imgs)

        if self.head_type == "regression":
            gt = batch["keypoints_norm"].to(device).float()
            loss = self.criterion(preds, gt)

        else:
            gt_hm = batch["heatmaps"].to(device).float() 
            if self.loss_type == "mse":
                loss = self.criterion(torch.sigmoid(preds), gt_hm)
            else:
                loss = self.criterion(preds, gt_hm)

        logs = {"train_loss" if train else "val_loss": loss}
        return loss, logs
    
    def training_step(self, batch, batch_idx):
        loss, log_dict = self._shared_step(batch, train=True)
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self._shared_step(batch, train=False)
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(config, "EPOCHS", 30))
        return [optimizer], [scheduler]
