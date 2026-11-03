import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from src.optics_alignment_regression.model import OpticsAlignmentRegressionModel
from src.optics_alignment_regression.dataset import load_h5_dataset

from configs.configs import (
    CHECKPOINT_DIR,
    ROOT_DIR,
    PROJECT_DIR,
    DATA_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS
)

def evaluate_model(h5_file_path, model_checkpoint_path):
    # Load the dataset from the HDF5 file
    dataset = load_h5_dataset(h5_file_path)
    images = dataset['image']
    x_offsets = dataset['x_offset']
    z_offsets = dataset['z_offset']

    # Combine x and z offsets into a single target array
    targets = torch.tensor(np.stack((x_offsets, z_offsets), axis=1), dtype=torch.float32)

    test_images, test_targets = images, targets
    # Convert images to PyTorch tensors and add channel dimension
    test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for test set
    test_dataset = torch.utils.data.TensorDataset(test_images, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model and load the checkpoint
    model = OpticsAlignmentRegressionModel(output_dim=2)
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()

    criterion = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for images_batch, targets_batch in test_loader:
            outputs = model(images_batch)
            loss = criterion(outputs, targets_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")