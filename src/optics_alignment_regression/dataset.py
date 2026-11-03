import fabio
import h5py

def load_h5_dataset(file_path):
    """
    Load a dataset from an HDF5 file.

    Parameters:
    - file_path (str): Path to the HDF5 file.

    Returns:
    - data (dict): A dictionary containing the datasets.
    """
    data = {}
    with h5py.File(file_path, 'r') as h5_file:
        for key in h5_file.keys():
            data[key] = h5_file[key][:]
    return data
