import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
import config
from dataset import FaceLandmarksDataset, get_files
from model import FaceAlignmentModel

def main(args):
    files = get_files(config.TRAIN_FOLDERS)
    train_size = int(len(files) * config.TRAIN_VAL_SPLIT)
    train_files, val_files = random_split(files, [train_size, len(files) - train_size])
    train_dataset = FaceLandmarksDataset(train_files, train=True)
    val_dataset = FaceLandmarksDataset(val_files, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = FaceAlignmentModel(model_type=args.model_type, loss_type=args.loss_type)

    logger = TensorBoardLogger(save_dir=str(config.LOG_DIR), name=args.experiment_name)
    
    trainer = pl.Trainer(max_epochs=config.EPOCHS, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    
    ckpt_path = Path(config.CHECKPOINT_DIR)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_path / f"{args.experiment_name}.ckpt"
    trainer.save_checkpoint(str(ckpt_file))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=config.MODEL_TYPE, choices=["resnet", "efficientnet"])
    parser.add_argument("--loss_type", type=str, default=config.LOSS_TYPE, choices=["mse", "wing", "adaptivewing"])
    parser.add_argument("--experiment_name", type=str, default="experiment")
    args = parser.parse_args()
    main(args)
