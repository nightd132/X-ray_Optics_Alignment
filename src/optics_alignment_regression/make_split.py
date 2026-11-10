import numpy as np
import h5py
from sklearn.model_selection import train_test_split

from src.optics_alignment_regression.dataset import load_h5_dataset
from configs.configs import H5_RAW_DATASET_PATH, DATA_DIR, TRAIN_VAL_TEST_INDICES_SPLIT_PATH

def split_dataset(h5_file_path, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets.

    Parameters:
    - h5_file_path (str): Path to the HDF5 file containing the dataset.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - train_images (np.ndarray): Training images.
    - test_images (np.ndarray): Test images.
    - train_targets (np.ndarray): Training targets (x and z offsets).
    - test_targets (np.ndarray): Test targets (x and z offsets).
    """
    dataset = load_h5_dataset(h5_file_path)
    images = dataset["images"]
    targets = dataset["offsets"]

    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    train_idx.sort(); test_idx.sort()

    return (images[train_idx], images[test_idx],
            targets[train_idx], targets[test_idx])

def save_split_dataset_as_h5(train_images, test_images, train_targets, test_targets, output_file_path):
    """
    Save the split dataset as an HDF5 file.

    Parameters:
    - train_images (np.ndarray): Training images.
    - test_images (np.ndarray): Test images.
    - train_targets (np.ndarray): Training targets (x and z offsets).
    - test_targets (np.ndarray): Test targets (x and z offsets).
    - output_file_path (str): Path to the output HDF5 file.
    """


    with h5py.File(output_file_path, 'w') as h5_file:
        # Create groups for training and test sets
        train_group = h5_file.create_group('train')
        test_group = h5_file.create_group('test')

        # Save training data
        train_group.create_dataset('image', data=train_images)
        train_group.create_dataset('targets', data=train_targets)

        # Save test data
        test_group.create_dataset('image', data=test_images)
        test_group.create_dataset('targets', data=test_targets)

def split_dataset_indices(data_path, save_path):
    with h5py.File(data_path, "r") as f:
        N = len(f.keys())
    
    idx = np.arange(N)

    trainval_idx, test_idx = train_test_split(idx, test_size=0.15, shuffle=False)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.15 / 0.85, shuffle=False)

    np.savez(save_path, train=train_idx, val=val_idx, test=test_idx)
    print(len(train_idx), len(val_idx), len(test_idx))

def main():
    # h5_file_path = H5_RAW_DATASET_PATH
    # print(f"Starting dataset split...")
    # train_images, test_images, train_targets, test_targets = split_dataset(h5_file_path)

    # print(f"Training images shape: {train_images.shape}")
    # print(f"Test images shape: {test_images.shape}")
    # print(f"Training targets shape: {train_targets.shape}")
    # print(f"Test targets shape: {test_targets.shape}")

    # print(f"Dataset split completed. Saving to HDF5 file...")
    # save_split_dataset_as_h5(train_images, test_images, train_targets, test_targets, output_file_path=f"{DATA_DIR}/split_dataset.h5")
    # print(f"Split dataset saved to {DATA_DIR}/split_dataset.h5")
    split_dataset_indices(H5_RAW_DATASET_PATH, TRAIN_VAL_TEST_INDICES_SPLIT_PATH)

if __name__ == "__main__":
    main()