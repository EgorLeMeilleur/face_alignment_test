# Face Alignment using Deep Learning

This project implements a face alignment algorithm for detecting 68 facial landmarks on a human face using modern deep learning techniques. The approach is built with PyTorch Lightning and leverages `timm` for model backbones.

## Report

`experiments_report.pdf` contains all the information about the experiments performed: the architectures used, all the results, etc.

## Requirements

Install all required packages via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages include:
*   `pytorch-lightning`
*   `timm`
*   `torch`
*   `torchvision`
*   `dlib`
*   `matplotlib`
*   `scikit-learn`
*   `pillow`
*   `numpy`
*   `timm`
*   `opencv-python`
*   `albumentations`

## Data

The training and testing data consist of the 300W and Menpo datasets. Place them in the following structure:

```bash
data/
├── 300W/
│   ├── train/
│   └── test/
└── Menpo/
    ├── train/
    └── test/
```

Annotations are provided in `.pts` format (IBUG format).

## Project Files

| File                  | Description                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `config.py`           | Contains hyperparameters and paths. You can change model, loss, head, etc.                       |
| `dataset.py`          | Custom dataset loader for images, annotations, preprocessing, augmentations.                     |
| `model.py`            | Lightning model definition using `timm` backbones, custom losses, heads.                         |
| `train.py`            | Script to train a model.                                                                         |
| `test.py`             | Script to test a trained model, compute CED curves, and AUC.                                     |
| `run_experiments.py`  | Script to run all experiments (train and test) in config sequentially.                           |
| `precompute_boxes.py` | Script to precompute dlib face boxes for dataset if dlib has been installed without gpu support. |

## Running Experiments

### Train a Model

```bash
python train.py --experiment_name "experiment1" --model_type "efficientnet_b0" --loss_type "MSE" --head_type "heatmap"
```

**Supported values:**

*   `--model_type`: `"efficientnet"`, `"convnext"`,
*   `--loss_type`: `"mse"`, `"wing"`, `"awing"`, `"focal"`, `"bce"`
*   `--head_type`: `"regression"`, `"heatmap"`

*All arguments have defaults set in `config.py`.*

### Test a Model

```bash
python test.py --experiment_name "experiment1" --model_type "efficientnet_b0" --loss_type "mse" --head_type "heatmap"
```
This will run the model on both 300W and Menpo test sets and generate evaluation metrics.

### Run All Experiments

```bash
python run_experiments.py
```
This will sequentially train and test all configured model/loss/head combinations.
```bash
python run_experiments.py --best
```
When the `--best` flag is used, the script will train and test only the best-performing experiment.

## Configuration

In `config.py`, you can configure:

*   `MODEL_TYPE`       # default backbone
*   `LOSS_TYPE`        # default loss function
*   `HEAD_TYPE`        # regression or heatmap
*   `BATCH_SIZE`       # batch size for training
*   `LEARNING_RATE`    # initial learning rate
*   `NUM_EPOCHS`       # total training epochs
*   `Dataset paths`    # paths to datasets and outputs
*   `Experiments`      # all experimets that will be conducted