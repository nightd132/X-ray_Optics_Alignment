import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from src.optics_alignment_regression.model import OpticsAlignmentRegressionModel
from src.optics_alignment_regression.dataset import load_split_h5, load_h5_dataset

from configs.configs import (
    CHECKPOINT_DIR,
    H5_DATASET_PATH,
    ROOT_DIR,
    PROJECT_DIR,
    TRAIN_TEST_SPLIT_PATH,
    TRAIN_VAL_LOSS_HISTORY_PATH,
    EVALUATION_RESULTS_PATH,
    DATA_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS
)

