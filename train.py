import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

import config
from dataset import FaceLandmarksDataset, get_files
from models import FaceAlignmentModel

def train(experiment_name, model_type, loss_type, head_type):
    files = get_files(config.TRAIN_FOLDERS)
    train_size = int(len(files) * config.TRAIN_VAL_SPLIT)
    train_files, val_files = random_split(files, [train_size, len(files) - train_size])
    train_dataset = FaceLandmarksDataset(train_files, train=True)
    val_dataset = FaceLandmarksDataset(val_files, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )


    model = FaceAlignmentModel(model_type=model_type, head_type=head_type, loss_type=loss_type)

    logger = TensorBoardLogger(save_dir=str(config.LOG_DIR), name=experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.CHECKPOINT_DIR),
        filename=experiment_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    trainer = pl.Trainer(max_epochs=config.EPOCHS, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=config.MODEL_TYPE, options=['efficientnet', 'convnext'])
    parser.add_argument("--loss_type", type=str, default=config.LOSS_TYPE, options=['mse', 'wing', 'awing', 'focal'])
    parser.add_argument("--head_type", type=str, default=config.HEAD_TYPE, options=['regression', 'heatmap'])
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()
    train(args.experiment_name, args.model_type, args.loss_type, args.head_type)
