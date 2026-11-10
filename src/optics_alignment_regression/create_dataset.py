from src.optics_alignment_regression.data_preprocessing import load_meta_data, load_fabio_image, extract_offsets, save_data_to_h5_file
import argparse
import h5py
from pathlib import Path
import re

from configs.configs import (
    RAW_DATA_DIR,
    H5_RAW_DATASET_PATH
)

def main():
    # regex to filter the path of data
    folder_re = re.compile(r"Mainz_\d{8}_\d{2}_AXO-Optics-Scan-forML_\d+W_Al-tape_noFlTu_SourceDet-\d+cm_detPos\d+")

    H5_RAW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_RAW_DATASET_PATH, 'w') as h5_file:
        idx = 0
        for folder in sorted(Path(RAW_DATA_DIR).glob("Mainz_*")):
            if not folder_re.match(folder.name):
                print(f"Skipping folder: {folder}")
                continue

            try:
                print(f"\nProcessing folder: {folder}")

                for image_path in sorted(folder.glob("image_*.cbf")):
                    image = load_fabio_image(image_path)

                    log_file_path = folder / f"log_{image_path.stem.split('_')[-1]}.txt"
                    meta_data = load_meta_data(log_file_path)
                    x_offset, z_offset = extract_offsets(meta_data)

                    if x_offset is None or z_offset is None:
                        print(f"Offsets not found for image: {image_path}")
                        continue

                    group_name = f"data_point_{idx}"
                    save_data_to_h5_file(h5_file, image, (x_offset, z_offset), group_name)
                    idx += 1

                    print(f"Saved image and offsets as group '{group_name}' to: {H5_RAW_DATASET_PATH}")

            except FileNotFoundError as e:
                print(f"File not found error while processing folder {folder}: {e}")

if __name__ == "__main__":
    main()