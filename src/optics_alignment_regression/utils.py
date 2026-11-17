import torch
from pathlib import Path
from src.optics_alignment_regression.model import OpticsAlignmentRegressionModel
import random
import numpy as np
import argparse
import pandas as pd

from configs.configs import (
    BIN_SIZE,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    WEIGHT_DECAY,
    NUM_WORKER,
    CHECKPOINT_DIR
)

def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = OpticsAlignmentRegressionModel(
        output_dim=cfg["output_dim"], input_dim=tuple(cfg["input_dim"])
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binsize", type=int, help="Which processed_bin{N}.h5 to train on", default=BIN_SIZE)
    parser.add_argument("--epoch", type=int, help="Set the number of epoch (int e.g 100)", default=EPOCHS)
    parser.add_argument("--lr", type=float, help="Set the learning rate (float e.g 0.01)", default=LEARNING_RATE)
    parser.add_argument("--batch", type=int, help="Set the batch size (int e.g 32)", default=BATCH_SIZE)
    parser.add_argument("--numworker", type=int, help="Set the number of workers (int e.g 4)", default=NUM_WORKER)
    parser.add_argument("--weightdecay", type=float, help="Set the weight-decay for AdamW (flaot e.g 0.01)", default=WEIGHT_DECAY)

    return parser.parse_args()

def find_best_experiment(csv_path):
    results = pd.read_csv(csv_path)
    best_experiment = results.loc[results["best_val_loss"].idxmin()]
    return best_experiment