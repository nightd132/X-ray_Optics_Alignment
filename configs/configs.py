from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR / "src" / "optics_alignment_regression"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = ROOT_DIR / "data" / "Mainz_2022-10_AXO-Optics-Scan-forML"
CHECKPOINT_DIR = ROOT_DIR/ "outputs" / "checkpoints"
H5_DATASET_PATH = DATA_DIR / "preprocessed_dataset.h5"
# Preprocessing parameters
BIN_SIZE = 4  # Size of the binning window for image preprocessing


# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50