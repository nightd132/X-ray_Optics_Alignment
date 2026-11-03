from pathlib import Path
import fabio, h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import re

from configs.configs import (
    CHECKPOINT_DIR,
    ROOT_DIR,
    PROJECT_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    H5_DATASET_PATH,
    BATCH_SIZE, 
    LEARNING_RATE,
    EPOCHS
)

def load_fabio_image(file_path):
    """
    Load an image using the fabio library.

    Parameters:
    - file_path (str): Path to the image file.

    Returns:
    - image (numpy.ndarray): The loaded image as a NumPy array.
    """
    image = fabio.open(file_path).data
    return image

def normalize_image(image):
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

def preprocess_image(image, bin_size):
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

def save_preprocessed_image_as_h5(h5_file, image, offsets, group_name):
    """
    Save the preprocessed image as an HDF5 file.
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

def find_brightest_point(image):
    """
    Find the coordinates of the brightest pixel in the image.

    Parameters:
    - image (numpy.ndarray): The input image.

    Returns:
    - x (int): The x-coordinate of the brightest pixel.
    - y (int): The y-coordinate of the brightest pixel.
    - max_intensity (float): The maximum intensity value.
    """
    x, y = np.unravel_index(np.argmax(image), image.shape)
    return x, y, float(image[x, y])

def find_optimal_image(folder):
    """
    Find the image with the highest single-pixel intensity.

    Parameters:
    - folder (str): Path to the folder containing CBF images.
    Returns:
    - best_path (Path): Path to the image with the highest intensity.
    - best_point (tuple): Coordinates (x, y) of the brightest pixel.
    - best_intensity (float): Maximum intensity value.
    """
    image_paths = sorted(Path(folder).glob("*.cbf"))

    if not image_paths:
        raise FileNotFoundError(f"No CBF images found in {folder}")

    best_path = None
    best_intensity = -np.inf
    best_point = None

    for image_path in image_paths:
        image = fabio.open(str(image_path)).data
        x, y, intensity = find_brightest_point(image)

        if intensity > best_intensity:
            best_path = image_path
            best_intensity = intensity
            best_point = (x, y)

    print("\n--- Highest maximum intensity ---")
    print(f"Image: {best_path}")
    print(f"Brightest pixel (x, y): {best_point}")
    print(f"Maximum intensity: {best_intensity}")

    return best_path, best_point, best_intensity

def load_log_txt_file(log_file_path):
    """
    Load the log text file and extract the hexpod x and hexpod z values.

    Parameters:
    - log_file_path (str): Path to the log text file.

    Returns:
    - hexapod_x (float): The hexapod x value.
    - hexapod_z (float): The hexapod z value.
    """
    pattern = re.compile(
        r"\b(hexapod)\s+([xz])\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )

    hexapod_x = None
    hexapod_z = None

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                axis = match.group(2).lower()
                value = float(match.group(3))
                if axis == 'x':
                    hexapod_x = value
                elif axis == 'z':
                    hexapod_z = value

    return hexapod_x, hexapod_z

def calculate_offsets_from_log(log_file_path, best_hexapod_x, best_hexapod_z):
    """
    Calculate the offsets from the log file based on the best hexapod x and z values.

    Parameters:
    - log_file_path (str): Path to the log text file.
    - best_hexapod_x (float): The best hexapod x value.
    - best_hexapod_z (float): The best hexapod z value.

    Returns:
    - x_offset (float): The calculated x offset.
    - z_offset (float): The calculated z offset.
    """
    hexapod_x, hexapod_z = load_log_txt_file(log_file_path)

    if hexapod_x is not None and hexapod_z is not None:
        # Calculate offsets relative to the best hexapod values
        x_offset = hexapod_x - best_hexapod_x
        z_offset = hexapod_z - best_hexapod_z

    return x_offset, z_offset

def main():
    
    folder_re = re.compile(r"Mainz_\d{8}_\d{2}_AXO-Optics-Scan-forML_\d+W_Al-tape_noFlTu_SourceDet-\d+cm_detPos\d+")

    H5_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_DATASET_PATH, 'w') as h5_file:
        idx = 0
        for folder in sorted(Path(RAW_DATA_DIR).glob("Mainz_*")):
            if not folder_re.match(folder.name):
                print(f"Skipping folder: {folder}")
                continue

            try:
                print(f"\nProcessing folder: {folder}")
                best_image_path, best_point, best_intensity = find_optimal_image(folder)
                print(f"Best image found: {best_image_path} with intensity {best_intensity} at point {best_point}")
                log_best_file_path = folder / f"log_{best_image_path.stem.split('_')[-1]}.txt"
                best_hexapod_x, best_hexapod_z = load_log_txt_file(log_best_file_path)

                if best_hexapod_x is None or best_hexapod_z is None:
                    print(f"Best hexapod x or z not found in log file: {log_best_file_path}")
                    continue

                for image_path in sorted(folder.glob("image_*.cbf")):
                    image = load_fabio_image(image_path)
                    preprocessed_image = preprocess_image(image, bin_size=4)
                    log_file_path = folder / f"log_{image_path.stem.split('_')[-1]}.txt"
                    x_offset, z_offset = calculate_offsets_from_log(log_file_path, best_hexapod_x, best_hexapod_z)

                    if x_offset is None or z_offset is None:
                        print(f"Offsets not found for image: {image_path}")
                        continue

                    group_name = f"data_point_{idx}"
                    save_preprocessed_image_as_h5(h5_file, preprocessed_image, (x_offset, z_offset), group_name)
                    idx += 1

                    print(f"Saved preprocessed image and offsets as group '{group_name}' to: {H5_DATASET_PATH}")

            except FileNotFoundError as e:
                print(f"File not found error while processing folder {folder}: {e}")

if __name__ == "__main__":
    main()