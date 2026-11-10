from pathlib import Path
import fabio, h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple
import re
from tqdm import tqdm

def load_fabio_image(file_path: str) -> np.ndarray:
    """
    Load an image using the fabio library.

    Parameters:
    - file_path (str): Path to the image file.

    Returns:
    - image (numpy.ndarray): The loaded image as a NumPy array.
    """
    image = fabio.open(file_path).data
    return image

def load_meta_data(file_path: str) -> Dict:
    """
    Load meta data from log.txt file.

    Parameters:
    - file_path (str): Path to the log.txt file.

    Returns:
    - dict: A dictionary containing all the meta data.
    """
    meta_data = {}
    with open(file_path) as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count > 9:
                line_attrs = line.split(" ")
                meta_data[line_attrs[0]+"_"+line_attrs[1]] = float(line_attrs[3])

    return meta_data

def extract_offsets(meta_data: Dict) -> Tuple[float, float]:
    """
    Extract x, z offsets from meta_data

    Parameter:
    - meta_data: dictionary containing all meta_data

    Returns:
    - tuple (float, float): x offset, z offset
    """
    return meta_data["hexapod_x"], meta_data["hexapod_z"]

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the range [0, 1].

    Parameters:
    - image (numpy.ndarray): The input image to normalize.

    Returns:
    - normalized_image (numpy.ndarray): The normalized image.
    """
    scaler = MinMaxScaler()
    normalized_image = scaler.fit_transform(image)
    return normalized_image

def preprocess_image(image: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Preprocess the image by binning it.

    Parameters:
    - image (numpy.ndarray): The input image to preprocess.
    - bin_size (int): The size of the binning window.

    Returns:
    - binned_scaled_image (numpy.ndarray): The binned and normalized image.
    """
    # Ensure the image dimensions are divisible by the bin size
    new_shape = (image.shape[0] // bin_size, bin_size, image.shape[1] // bin_size, bin_size)
    binned_image = image[:new_shape[0]*bin_size, :new_shape[2]*bin_size].reshape(new_shape).mean(axis=(1, 3))
    binned_scaled_image = normalize_image(binned_image)
    return binned_scaled_image

def save_data_to_h5_file(h5_file, image, offsets, group_name):
    """
    Save the data point including image and offsets as an HDF5 file.
    Group the image and (x, z) offsets in the HDF5 file.
    Create groups for each data point (0....n) and then in each group having X -> image (as 2d numpy) and Y -> targets.

    Parameters:
    - h5_file (h5py.File): The HDF5 file object to save the data.
    - image (numpy.ndarray): The preprocessed image to save.
    - offsets (tuple): A tuple containing the x and z offsets.
    - group_name (str): Name of the group to create in the HDF5 file.
    """
    grp = h5_file.create_group(group_name)
    grp.create_dataset('image', data=image, compression='gzip')
    grp.create_dataset('x_offset', data=offsets[0])
    grp.create_dataset('z_offset', data=offsets[1])

def make_processed(raw_path, out_path, binsize):
    with h5py.File(raw_path, "r") as fin:
        N = sum(k.startswith("data_point_") for k in fin.keys())

        # shape from the first sample
        x0 = preprocess_image(fin["data_point_0"]["image"][()].astype(np.float32), binsize)

        with h5py.File(out_path, "w") as fout:
            images = fout.create_dataset("images", (N, *x0.shape), dtype="float32")
            offsets = fout.create_dataset("offsets", (N, 2), dtype="float32")
            fout.attrs["binsize"] = binsize

            for j in tqdm(range(N)):
                g = fin[f"data_point_{j}"]
                images[j] = preprocess_image(g["image"][()].astype(np.float32), binsize)
                offsets[j] = (g["x_offset"][()], g["z_offset"][()])