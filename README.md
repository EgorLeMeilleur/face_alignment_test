# Face Alignment using Deep Learning

This project implements a face alignment algorithm for detecting 68 facial landmarks on a human face using modern deep learning techniques. The approach is built with PyTorch Lightning and leverages timm for model backbones.

## Overview

The following experiments are included:
- **ResNet + MSE Loss**
- **EfficientNet + Wing Loss**
- **EfficientNet + Adaptive Wing Loss**

## Data

The training and testing data consist of the 300W and Menpo datasets, organized as follows:

data/ 300W/ train/ test/ Menpo/ train/ test/


Annotations are provided in `.pts` format (IBUG format).

## Setup

Required packages include: pytorch-lightning, timm, torch, torchvision, dlib, fpdf, matplotlib, and scikit-learn.

    Data:
    Download and place the 300W and Menpo datasets in the data directory.

Files

    config.py: Contains hyperparameters and paths.
    dataset.py: Custom dataset loader that reads images, annotations, and applies augmentations.
    model.py: Contains the model definition using timm and the custom Wing loss.
    train.py: Training script with TensorBoard logging.
    test.py: Testing script that computes CED curves (and AUC) for 300W and Menpo.
    run_experiments.py: Script to run all experiments sequentially.
    report_generation.py: Script to generate a PDF report with experiment summaries and CED graphs.
    README.md: This documentation file.

Running Experiments

To run all experiments, execute:

    python run_experiments.py

This script trains and tests each experiment variant sequentially.
After experiments are complete and the CED graphs have been generated, create a PDF report will be created
The report will be saved in the outputs/reports directory.

TensorBoard

To visualize training logs, run:

    tensorboard --logdir outputs/logs