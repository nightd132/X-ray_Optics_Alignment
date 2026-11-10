import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from pathlib import Path

from src.optics_alignment_regression.model import OpticsAlignmentRegressionModel
from src.optics_alignment_regression.dataset import X_Ray_Optics_Dataset

from configs.configs import (
    CHECKPOINT_DIR,
    H5_RAW_DATASET_PATH,
    TRAIN_VAL_TEST_INDICES_SPLIT_PATH,
    DATA_DIR,
    NUM_WORKER,
    WEIGHT_DECAY,
    BIN_SIZE,
    BATCH_SIZE, 
    LEARNING_RATE,
    EPOCHS
)

def build_data_loaders(args):
    split_idx = np.load(TRAIN_VAL_TEST_INDICES_SPLIT_PATH)
    train_idx = split_idx["train"]
    val_idx = split_idx["val"]
    processed_data_path = DATA_DIR/ f"processed_bin{args.binsize}.h5"
    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"{processed_data_path} not found. Run: python scripts/make_processed.py --binsize {args.binsize}"
        )
    train_ds = X_Ray_Optics_Dataset(processed_data_path, train_idx)
    val_ds = X_Ray_Optics_Dataset(processed_data_path, val_idx)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                          num_workers=args.numworker, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.numworker)
    return train_dl, val_dl

def save_checkpoint(path, model, epoch, val_loss, config, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = OpticsAlignmentRegressionModel(
        output_dim=cfg["output_dim"], input_dim=tuple(cfg["input_dim"])
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, offsets in loader:
        images, offsets = images.to(device), offsets.to(device)
        optimizer.zero_grad()
        output_offsets = model(images)
        loss = criterion(output_offsets, offsets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    for images, offsets in loader:
        images, offsets = images.to(device), offsets.to(device)
        output_offsets = model(images)
        loss = criterion(output_offsets, offsets)
        eval_loss += loss.item() * images.size(0)
    return eval_loss/ len(loader.dataset)


def train_model(model, train_dl, val_dl, criterion, optimizer, device, epochs, ckpt_path, config):

    ckpt_path.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    # Training loop
    print("Starting training...")
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        avg_train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        train_hist.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")

        # Validation loop
        print("Starting validation...")

        avg_val_loss = evaluate(model, val_dl, criterion, device)
        val_hist.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            # Save the best model checkpoint
            save_checkpoint(ckpt_path / "best.pt", model, epoch, best_val, config, optimizer)
            print(f"Best model saved with validation loss: {best_val:.6f}")

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        save_checkpoint(ckpt_path / "last.pt", model, epoch, avg_val_loss, config, optimizer)
        np.savez(ckpt_path/"train_val_loss_history.npz", train_loss=train_hist, val_loss=val_hist)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binsize", type=int, help="Which processed_bin{N}.h5 to train on", default=BIN_SIZE)
    parser.add_argument("--epoch", type=int, help="Set the number of epoch (int e.g 100)", default=EPOCHS)
    parser.add_argument("--lr", type=float, help="Set the learning rate (float e.g 0.01)", default=LEARNING_RATE)
    parser.add_argument("--batch", type=int, help="Set the batch size (int e.g 32)", default=BATCH_SIZE)
    parser.add_argument("--numworker", type=int, help="Set the number of workers (int e.g 4)", default=NUM_WORKER)
    parser.add_argument("--weightdecay", type=float, help="Set the weight-decay for AdamW (flaot e.g 0.01)", default=WEIGHT_DECAY)

    return parser.parse_args()


def main():
    # Parse arguments from CLI
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build data loader
    train_dl, val_dl = build_data_loaders(args)
    x0, _ = train_dl.dataset[0]

    # Initialize the model, loss function, and optimizer
    model = OpticsAlignmentRegressionModel(output_dim=2, input_dim=tuple(x0.shape)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    # Create checkpoint path with correspond setup
    ckpt_path = CHECKPOINT_DIR /f"binsize_{args.binsize}_epoch_{args.epoch}_lr_{args.lr}_batch_{args.batch}_weightdecay_{args.weightdecay}"

    config = dict(binsize=args.binsize, input_dim=list(x0.shape), output_dim=2, 
                  lr=args.lr, batch=args.batch, weightdecay=args.weightdecay)
    
    # training model
    train_model(model, train_dl, val_dl, criterion, optimizer, device, args.epoch, ckpt_path, config)

if __name__ == "__main__":
    main()