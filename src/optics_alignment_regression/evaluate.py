import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import csv
from tqdm import tqdm

from src.optics_alignment_regression.model import OpticsAlignmentRegressionModel
from src.optics_alignment_regression.dataset import X_Ray_Optics_Dataset

from configs.configs import (
    CHECKPOINT_DIR,
    TRAIN_VAL_TEST_INDICES_SPLIT_PATH,
    DATA_DIR,
    NUM_WORKER,
    WEIGHT_DECAY,
    BIN_SIZE,
    BATCH_SIZE, 
    LEARNING_RATE,
    EPOCHS,
    EVALUATION_RESULTS_PATH
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

def build_data_loaders(args):
    split_idx = np.load(TRAIN_VAL_TEST_INDICES_SPLIT_PATH)
    test_idx = split_idx["test"]
    processed_data_path = DATA_DIR/ f"processed_bin{args.binsize}.h5"
    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"{processed_data_path} not found. Run: python scripts/make_processed.py --binsize {args.binsize}"
        )
    test_ds = X_Ray_Optics_Dataset(processed_data_path, test_idx)

    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.numworker)
    return test_dl

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    progress_bar = tqdm(loader, desc="Test")
    eval_loss = 0.0
    for images, offsets in progress_bar:
        images, offsets = images.to(device), offsets.to(device)
        output_offsets = model(images)
        loss = criterion(output_offsets, offsets)
        eval_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(
            loss = f"{eval_loss:.6f}"
        )

    return eval_loss/ len(loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binsize", type=int, help="Which processed_bin{N}.h5 to train on", default=BIN_SIZE)
    parser.add_argument("--epoch", type=int, help="Set the number of epoch (int e.g 100)", default=EPOCHS)
    parser.add_argument("--lr", type=float, help="Set the learning rate (float e.g 0.01)", default=LEARNING_RATE)
    parser.add_argument("--batch", type=int, help="Set the batch size (int e.g 32)", default=BATCH_SIZE)
    parser.add_argument("--numworker", type=int, help="Set the number of workers (int e.g 4)", default=NUM_WORKER)
    parser.add_argument("--weightdecay", type=float, help="Set the weight-decay for AdamW (flaot e.g 0.01)", default=WEIGHT_DECAY)

    return parser.parse_args()

def save_evaluation_results(loss, args, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "binsize": args.binsize,
        "epochs": args.epoch,
        "learning_rate": args.lr,
        "batch_size": args.batch,
        "weight_decay": args.weightdecay,
        "test_loss": loss,
    }

    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)
    


def main():
    # Parse arguments from CLI
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build data loader
    test_dl = build_data_loaders(args)
    
    # Create checkpoint path with correspond setup
    ckpt_path = CHECKPOINT_DIR /f"binsize_{args.binsize}_epoch_{args.epoch}_lr_{args.lr}_batch_{args.batch}_weightdecay_{args.weightdecay}"
    if not ckpt_path.exists:
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run: python scripts/make_processed.py --binsize {args.binsize}\n And python scripts/train.py --binsize {args.binsize} --epoch {args.epoch} --lr {args.lr} --weightdecay {args.weig}"
        )
    # Initialize the model, loss function, and optimizer
    model, cfg = load_checkpoint(ckpt_path/ "best.pt", device)
    criterion = nn.MSELoss()
    
    # evaluate model
    test_loss = evaluate(model, test_dl, criterion, device)
    save_evaluation_results(test_loss, args, EVALUATION_RESULTS_PATH)


if __name__ == "__main__":
    main()