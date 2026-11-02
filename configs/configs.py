from pathlib import Path
# Paths

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_DIR = ROOT_DIR / "src" / "optics_alignment_regression"
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR/ "outputs" / "checkpoints"


# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50