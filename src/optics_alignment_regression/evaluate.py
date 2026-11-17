import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import csv
from tqdm import tqdm
import random
import pandas as pd

from src.optics_alignment_regression.model import OpticsAlignmentRegressionModel
from src.optics_alignment_regression.dataset import X_Ray_Optics_Dataset
from src.optics_alignment_regression.utils import seed_everything, load_checkpoint, find_best_experiment

from configs.configs import (
    CHECKPOINT_DIR,
    TRAIN_VAL_TEST_INDICES_SPLIT_PATH,
    DATA_DIR,
    NUM_WORKER,
    EVALUATION_RESULTS_PATH,
    EXPERIMENT_SUMMARY_CSV_PATH
)

def build_data_loaders(experiment):

    split_idx = np.load(TRAIN_VAL_TEST_INDICES_SPLIT_PATH)
    test_idx = split_idx["test"]

    binsize = int(experiment["binsize"])
    batch_size = int(experiment["batch_size"])

    processed_data_path = DATA_DIR / f"processed_bin{binsize}.h5"

    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"{processed_data_path} not found."
        )

    test_ds = X_Ray_Optics_Dataset(
        processed_data_path,
        test_idx
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKER
    )

    return test_dl

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    progress_bar = tqdm(loader, desc="Test")
    eval_loss = 0.0
    preds, targets = [], []
    for images, offsets in progress_bar:
        images, offsets = images.to(device), offsets.to(device)
        output_offsets = model(images)
        loss = criterion(output_offsets, offsets)
        eval_loss += loss.item() * images.size(0)
        preds.append(output_offsets.cpu().numpy())
        targets.append(offsets.cpu().numpy())
        progress_bar.set_postfix(
            loss = f"{eval_loss:.6f}"
        )
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return eval_loss/ len(loader.dataset), preds, targets

def save_evaluation_results( experiment, test_loss, preds, targets, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    errors = preds - targets
    
    results = {
        "experiment": experiment["experiment"],
        "binsize": int(experiment["binsize"]),
        "epochs": int(experiment["epochs"]),
        "best_epoch": int(experiment["best_epoch"]),
        "learning_rate": float(experiment["learning_rate"]),
        "batch_size": int(experiment["batch_size"]),
        "weight_decay": float(experiment["weight_decay"]),
        "best_val_loss": float(experiment["best_val_loss"]),
        "test_loss": test_loss,
        "test_mae_x": np.abs(errors[:, 0]).mean(),
        "test_mae_z": np.abs(errors[:, 1]).mean(),
        "test_rmse_x": np.sqrt(np.mean(errors[:, 0] ** 2)),
        "test_rmse_z": np.sqrt(np.mean(errors[:, 1] ** 2)),
    }

    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter( f, fieldnames=results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)

def main():
    # Set random seed for reproducibility
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build data loader
    best_experiment = find_best_experiment(EXPERIMENT_SUMMARY_CSV_PATH)

    ckpt_path = CHECKPOINT_DIR / best_experiment["experiment"]
    test_dl = build_data_loaders(best_experiment)
    
    # Create checkpoint path with correspond setup
    # Initialize the model, loss function, and optimizer
    model, cfg = load_checkpoint(ckpt_path/ "best.pt", device)
    criterion = nn.MSELoss()
    
    # evaluate model
    test_loss, preds, targets = evaluate(model, test_dl, criterion, device)
    save_evaluation_results(best_experiment, test_loss, preds, targets, EVALUATION_RESULTS_PATH)
    df = pd.DataFrame({
        "target_x": targets[:, 0],
        "target_z": targets[:, 1],
        "pred_x":   preds[:, 0],
        "pred_z":   preds[:, 1],
    })
    df["err_x"] = df["pred_x"] - df["target_x"]
    df["err_z"] = df["pred_z"] - df["target_z"]

    df.to_csv(ckpt_path / "test_predictions.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()