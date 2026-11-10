import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split

from src.optics_alignment_regression.model import OpticsAlignmentRegressionModel
from src.optics_alignment_regression.dataset import load_h5_dataset, load_split_h5

from configs.configs import (
    CHECKPOINT_DIR,
    ROOT_DIR,
    PROJECT_DIR,
    DATA_DIR,
    H5_RAW_DATASET_PATH,
    TRAIN_TEST_SPLIT_PATH,
    TRAIN_VAL_LOSS_HISTORY_PATH,
    EVALUATION_RESULTS_PATH,
    BATCH_SIZE, 
    LEARNING_RATE,
    EPOCHS
)

def train_model(h5_file_path, device):

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataset from the HDF5 file
    training_set = load_split_h5(h5_file_path, "train")

    images = training_set[0]
    targets = training_set[1]
    targets = np.asarray(targets, dtype=np.float32)

    # Split the dataset into training and validation sets
    train_images, val_images, train_targets, val_targets = train_test_split(
        images, targets, test_size=0.2, random_state=42
    )

    # Convert images to PyTorch tensors and add channel dimension
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
    val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)

    # Create DataLoader for training and validation sets
    train_dataset = torch.utils.data.TensorDataset(train_images, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_images, val_targets)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = OpticsAlignmentRegressionModel(output_dim=2, input_dim=(1, 86, 81)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")

    # Training loop
    print("Starting training...")
    train_hist, val_hist = [], []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        for images_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            images_batch, targets_batch = images_batch.to(device), targets_batch.to(device)
            outputs = model(images_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Batch Loss: {loss.item():.6f}")

        avg_train_loss = running_loss / len(train_loader)
        train_hist.append(avg_train_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}")

        # Validation loop
        print("Starting validation...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_batch, targets_batch in val_loader:
                images_batch, targets_batch = images_batch.to(device), targets_batch.to(device)
                outputs = model(images_batch)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item()
                print(f"Validation Batch Loss: {loss.item():.6f}")

        avg_val_loss = val_loss / len(val_loader)
        val_hist.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            # Save the best model checkpoint
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"Best model saved with validation loss: {best_val:.6f}")

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
    np.savez(TRAIN_VAL_LOSS_HISTORY_PATH, train_loss=train_hist, val_loss=val_hist)

def main():
    torch.manual_seed(42)  # Set random seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train the model using the preprocessed dataset
    train_model(h5_file_path=DATA_DIR / "split_dataset.h5", device=device)

if __name__ == "__main__":
    main()