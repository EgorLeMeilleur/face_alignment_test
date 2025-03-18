from pathlib import Path

DATA_DIR = Path("data")
THREE_HUNDRED_W_DIR = DATA_DIR / "300W"
MENPO_DIR = DATA_DIR / "Menpo"

TRAIN_FOLDERS = [THREE_HUNDRED_W_DIR / "train", MENPO_DIR / "train"]

TEST_FOLDERS = {
    "300W": THREE_HUNDRED_W_DIR / "test",
    "Menpo": MENPO_DIR / "test"
}

BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
EPOCHS = 5
IMAGE_SIZE = (64, 64)
NUM_POINTS = 68
TRAIN_VAL_SPLIT = 0.8
CROP_EXPANSION = 20
MAX_ERROR_THRESHOLD = 0.08

LOSS_TYPE = "mse"
MODEL_TYPE = "resnet"

EXPERIMENTS = [
    {"name": "resnet_mse", "model_type": "resnet", "loss_type": "mse"},
    {"name": "resnet_wing", "model_type": "resnet", "loss_type": "wing"},
    {"name": "efficientnet_mse", "model_type": "efficientnet", "loss_type": "wing"},
    {"name": "efficientnet_wing", "model_type": "efficientnet", "loss_type": "adaptivewing"},
]

OUTPUT_DIR = Path("outputs")
LOG_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
REPORT_DIR = OUTPUT_DIR / "reports"
