from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT_DIR / "src" / "optics_alignment_regression"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = ROOT_DIR / "data" / "Mainz_2022-10_AXO-Optics-Scan-forML"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
H5_RAW_DATASET_PATH = DATA_DIR / "raw_dataset.h5"
TRAIN_VAL_TEST_INDICES_SPLIT_PATH = DATA_DIR/ "split.npz"
TRAIN_TEST_SPLIT_PATH = DATA_DIR / "split_dataset.h5"
EVALUATION_RESULTS_PATH = OUTPUT_DIR / "evaluation_results.csv"

# Preprocessing parameters
BIN_SIZE = 12  # Size of the binning window for image preprocessing


# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
NUM_WORKER = 4
WEIGHT_DECAY = 0.01