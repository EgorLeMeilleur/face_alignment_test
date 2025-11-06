from pathlib import Path

DATA_DIR = Path("data")
THREE_HUNDRED_W_DIR = DATA_DIR / "300W"
MENPO_DIR = DATA_DIR / "Menpo"

TRAIN_FOLDERS = [THREE_HUNDRED_W_DIR / "train", MENPO_DIR / "train"]

TEST_FOLDERS = {
    "300W": THREE_HUNDRED_W_DIR / "test",
    "Menpo": MENPO_DIR / "test"
}

NUM_WORKERS = 10
PRECOMPUTE = "precomputed_face_boxes.json"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_POINTS = 68
TRAIN_VAL_SPLIT = 0.8
CROP_EXPANSION = 0.1
MAX_ERROR_THRESHOLD = 0.08
HEATMAP_SIZE = 64
SIGMA = 2.0
HEATMAP_POS_WEIGHT = 5.0

EXPERIMENTS = [
    {"name": "efficientnet_mse_regression", "model_type": "efficientnet", "loss_type": "mse", "head_type": "regression"},
    {"name": "efficientnet_wing_regression", "model_type": "efficientnet", "loss_type": "wing", "head_type": "regression"},
    {"name": "efficientnet_awing_regression", "model_type": "efficientnet", "loss_type": "awing", "head_type": "regression"},
    {"name": "efficientnet_mse_heatmap", "model_type": "efficientnet", "loss_type": "mse", "head_type": "heatmap"},
    {"name": "efficientnet_focal_heatmap", "model_type": "efficientnet", "loss_type": "focal", "head_type": "heatmap"},
    {"name": "efficientnet_bce_heatmap", "model_type": "efficientnet", "loss_type": "bce", "head_type": "heatmap"},
    # {"name": "convnext_mse_regression", "model_type": "convnext", "loss_type": "mse", "head_type": "regression"},
    # {"name": "convnext_wing_regression", "model_type": "convnext", "loss_type": "wing", "head_type": "regression"},
    # {"name": "convnext_awing_regression", "model_type": "convnext", "loss_type": "awing", "head_type": "regression"},
    # {"name": "convnext_mse_heatmap", "model_type": "convnext", "loss_type": "mse", "head_type": "heatmap"},
    # {"name": "convnext_focal_heatmap", "model_type": "convnext", "loss_type": "focal", "head_type": "heatmap"},
    # {"name": "convnext_bce_heatmap", "model_type": "convnext", "loss_type": "bce", "head_type": "heatmap"},
]

MODEL_TYPE = "efficientnet"
LOSS_TYPE = "mse"
HEAD_TYPE = "regression"

OUTPUT_DIR = Path("outputs")
LOG_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
REPORT_DIR = OUTPUT_DIR / "reports"