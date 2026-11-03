import h5py
import numpy as np

def load_h5_dataset(file_path):
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

def load_split_h5(path, split):
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