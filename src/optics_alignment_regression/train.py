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

def train_model(h5_file_path):
    # Load the dataset from the HDF5 file
    dataset = load_h5_dataset(h5_file_path)
    images = dataset['image']
    x_offsets = dataset['x_offset']
    z_offsets = dataset['z_offset']

    # Combine x and z offsets into a single target array
    targets = torch.tensor(np.stack((x_offsets, z_offsets), axis=1), dtype=torch.float32)

    # Split the dataset into training and validation sets
    train_images, val_images, train_targets, val_targets = train_test_split(
        images, targets, test_size=0.2, random_state=42
    )

    # Convert images to PyTorch tensors and add channel dimension
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
    val_images = torch.tensor(val_images, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for training and validation sets
    train_dataset = torch.utils.data.TensorDataset(train_images, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_images, val_targets)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = OpticsAlignmentRegressionModel(output_dim=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_batch, targets_batch in val_loader:
                outputs = model(images_batch)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")