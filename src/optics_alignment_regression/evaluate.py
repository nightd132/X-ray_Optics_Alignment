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

def evaluate_model(h5_file_path, model_checkpoint_path, device):
    # Load test dataset from the HDF5 file
    test_set = load_split_h5(h5_file_path, "test")
    images = test_set[0]
    targets = test_set[1]
    targets = np.asarray(targets, dtype=np.float32)

    test_images, test_targets = images, targets
    # Convert images to PyTorch tensors and add channel dimension
    test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)

    # Create DataLoader for test set
    test_dataset = torch.utils.data.TensorDataset(test_images, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model and load the checkpoint
    model = OpticsAlignmentRegressionModel(output_dim=2, input_dim=(1, 86, 81)).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()

    criterion = nn.MSELoss()
    test_loss = 0.0
    print("Starting evaluation on test set...")
    preds = []
    with torch.no_grad():
        for images_batch, targets_batch in test_loader:
            images_batch, targets_batch = images_batch.to(device), targets_batch.to(device)
            outputs = model(images_batch)
            loss = criterion(outputs, targets_batch)
            test_loss += loss.item()
            print(f"Batch Test Loss: {loss.item():.6f}")
            preds.extend(outputs.cpu().numpy())
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    np.savez(EVALUATION_RESULTS_PATH, test_loss=avg_test_loss, predictions=preds, targets=test_targets.cpu().numpy())

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Evaluate the model using the preprocessed dataset and the trained model checkpoint
    evaluate_model(h5_file_path=DATA_DIR / "split_dataset.h5", model_checkpoint_path=CHECKPOINT_DIR / "best_model.pth", device=device)

if __name__ == "__main__":
    main()