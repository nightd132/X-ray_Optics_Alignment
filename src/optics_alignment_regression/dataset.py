import h5py
import numpy as np
from typing import Dict
import torch
from src.optics_alignment_regression.data_preprocessing import preprocess_image
from torch.utils.data import Dataset

def load_h5_dataset(file_path: str) -> Dict:
    """
    Load all data points into numpy arrays.
    
    Parameters:
    - file_path (str): Path to the HDF5 file.
    
    Returns:
    - dict: A dictionary containing the loaded data.
    """
    with h5py.File(file_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
        first = f[keys[0]]["image"]
        images = np.empty((len(keys), *first.shape), dtype=first.dtype)
        offsets = np.empty((len(keys), 2), dtype=np.float32)
        for i, k in enumerate(keys):
            images[i] = f[k]["image"][()]
            offsets[i] = (f[k]["x_offset"][()], f[k]["z_offset"][()])
    return {"keys": keys, "images": images, "offsets": offsets}

def load_split_h5(path: str, split: str):
    """
    Load a specific split (train or test) from the HDF5 file.

    Parameters:
    - path (str): Path to the HDF5 file.
    - split (str): The split to load ("train" or "test").
    
    Returns:
    - tuple: A tuple containing the images and targets for the specified split.
    """
    with h5py.File(path, "r") as f:
        return f[split]["image"][()], f[split]["targets"][()]

class X_Ray_Optics_Dataset(Dataset):
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = indices
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __get_item__(self, i):
        if self.file == None:
            self.file = h5py.open(self.h5_path, "r")

        
        group = self.file[f"data_point_{self.indices[i]}"]
        image = group["image"][()].astype(np.float32)
        preprocessed_image = preprocess_image(image)
        offsets = np.array([g["x_offset"][()], g["z_offset"][()]], dtype=np.float32)

        return preprocessed_image, offsets
