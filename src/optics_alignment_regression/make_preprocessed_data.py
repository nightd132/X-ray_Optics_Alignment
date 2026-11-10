import h5py
from src.optics_alignment_regression.data_preprocessing import make_processed
import argparse

from configs.configs import (
    BIN_SIZE,
    H5_RAW_DATASET_PATH,
    DATA_DIR
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binsize", type=int, default=BIN_SIZE)
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"processed_bin{args.binsize}.h5"
    print("Start make processed dataset ...")
    make_processed(H5_RAW_DATASET_PATH, out, args.binsize)
    print(f"Done. Saved: {out}")

if __name__ == "__main__":
    main()