import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import config
from losses import WingLoss, AdaptiveWingLoss, HeatmapFocalLoss

class FaceAlignmentModel(pl.LightningModule):
    def __init__(self, backbone_name: str = "efficientnet_b0", head_type: str = "regression", loss_type: str = "mse"):
        super().__init__()
        self.save_hyperparameters()

        self.num_points = config.NUM_POINTS
        self.img_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        self.img_h, self.img_w = int(self.img_size[0]), int(self.img_size[1])
        hm_def = (config.HEATMAP_SIZE, config.HEATMAP_SIZE)
        self.hm_h, self.hm_w = int(hm_def[0]), int(hm_def[1])
        self.lr = 1e-3

        self.head_type = head_type
        self.loss_type = loss_type
        self.backbone_name = backbone_name

        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True, in_chans=3)
        feature_info = self.backbone.feature_info
        last_info = feature_info.info_list[-1]
        in_channels = last_info['num_chs']

        if self.head_type == "regression":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels, in_channels // 2 if in_channels >= 64 else in_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(in_channels // 2 if in_channels >= 64 else in_channels, self.num_points * 2)
            )
            if loss_type == "mse":
                self.criterion = nn.MSELoss(reduction='none')
            elif loss_type == "wing":
                self.criterion = WingLoss()
            elif loss_type == "adaptive_wing":
                self.criterion = AdaptiveWingLoss()

        elif self.head_type == "heatmap":
            self.upsample_head = self._make_upsample_head(in_channels, self.num_points, target_size=(self.hm_h, self.hm_w))
            if loss_type == "mse":
                self.criterion = nn.MSELoss()
            elif loss_type == "focal":
                self.criterion = HeatmapFocalLoss()
            else:
                raise ValueError(f"Unsupported loss_type for heatmap: {loss_type}")
        else:
            raise ValueError("Unsupported head_type: choose 'regression' or 'heatmap'")

    def _make_upsample_head(self, in_ch, out_ch, target_size=(64,64)):
        layers = []
        layers.append(nn.Conv2d(in_ch, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(2):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, out_ch, kernel_size=1, bias=True))
        head = nn.Sequential(*layers)
        return head

    def forward(self, x):
        if self.head_type == "regression":
            feats = self.backbone(x)
            pooled = self.pool(feats)
            out = self.regressor(pooled)
            out = out.view(-1, self.num_points, 2)
            return out
        else:
            feats = self.backbone(x)
            logits = self.upsample_head(feats)
            logits = torch.nn.functional.interpolate(logits, size=(self.hm_h, self.hm_w), mode='bilinear', align_corners=False)
            return logits

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
