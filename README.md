# X-ray_Optics_Alignment

## 1. Introduction

### Experiment

 **Real-Time 2D X-Ray Diffraction (2D-XRD)**
- Purpose is to investigate the crystal structure of a material and how that structure changes during a reaction.
- For example, researchers can use this setup to observe:
- The formation of new crystalline phases during a chemical reaction.
- Changes in crystal structure as temperature increases.
- Changes in crystallinity and lattice parameters over time.
- The formation or disappearance of crystalline compounds.
- The main advantage is that researchers can monitor these changes in situ
- The basic principle is:
    - X-ray source → X-ray optics → Sample → 2D detector → Diffraction pattern → Structural information

### Problem
- **Dataset**: (~60,000)
- **Goal**: Detector Image → offset to optimal position in (X, Z)
- **Idea**: Train network to directly predict offsets


## 2. Dataset Handling

Script: `src/optics_alignment_regression/create_dataset.py`. Output: `data/raw_dataset.h5`

### Labels
- For each scan folder (one detector distance + camera position), the **optimal position** are [0,0].
- The hexapod (x, z) of that image, read from `log_<id>.txt`, is the reference.
- Each image's target is `offset = hexapod_pos(image) - [0, 0]`.

Units: same as the hexapod log. Sign: positive = image is displaced in +x / +z from the optimum. The network predicts this offset; to correct, move the hexapod by `-offset`.

### HDF5 layout
```data_point_<N>/
  image     float32,
  x_offset  float64, scalar
  z_offset  float64, scalar
```

### Usage
```bash
    PYTHONPATH="$PWD" python src/optics_alignment_regression/create_dataset.py
```

## 3. Data Preprocessing

Script: `src/optics_alignment_regression/create_dataset.py`. Output: `data/processed_bin{N}.h5`

### Preprocessing
- Binning: 4x4 mean, edges cropped to a multiple of 4.
- Invalid pixels set to 0 before binning.
- Normalisation is stored (all pixels in range [0, 1]).

### HDF5 layout
```data_point_<N>/
  image     float32, # H/N x W/N
  x_offset  float64, scalar
  z_offset  float64, scalar
```

### Usage
```bash
    PYTHONPATH="$PWD" python src/optics_alignment_regression/make_preprocessed_data.py --binsize N
```

replace N with a number by default N = 12.

## 4. Data Splitting

Script: `src/optics_alignment_regression/make_split.py`. Output: `data/split.npz`

- Split the preprocessed_dataset into train, validation and test set with proportion 70:15:15.
- Store as split.npz
- Here we just split and store the indices to reduce the storage.

### Split file layout
```"train" : [0, 1, ..., 0.7*N-1]
"val": [0.7*N, ..., 0.85*N -1]
"test":[0.85*N, ..., N]
```
  
### Usage
```bash
    PYTHONPATH="$PWD" python src/optics_alignment_regression/make_split.py
```

## 5. Model

### Idea
- We got Image as input so we need model to turn image to matrix -> CNN (here we use 3 layer with max pooling)
- Then we flatten it to a vector and then a full connected network.
- We have x, z as output both are continuous value -> Linear layer (simple 2 layer with ReLU activation function)

## 6. Training

Script: `src/optics_alignment_regression/train.py`. Output: `output/checkpoint/*`

- A normal training neural network process
- Using AdamW as optimiser: its best for deep networks (could use another and compare later)
- output include folder name depend on setup (e.g binsize_12_epoch_100_lr_0.1_batch_64_weightdecay_0.01) and contain
  - **best.pt**: contain best model + config
  - **last.pt**: contain last model + config, use for resume if the training was broken
  - **train_val_loss_history.csv**: contain the average losses per epoch
- Default setup if you do not use the command-line option:
  - BIN_SIZE = 12 --binsize
  - BATCH_SIZE = 64 --batch
  - LEARNING_RATE = 0.001 --lr
  - EPOCHS = 100 --epoch
  - NUM_WORKER = 4 --numworker
  - WEIGHT_DECAY = 0.01 --weightdecay

### Usage
Here is 4 set up we've run
```bash
    PYTHONPATH="$PWD" python src/optics_alignment_regression/train.py --lr 0.1
    PYTHONPATH="$PWD" python src/optics_alignment_regression/train.py --lr 0.01
    PYTHONPATH="$PWD" python src/optics_alignment_regression/train.py --lr 0.001
    PYTHONPATH="$PWD" python src/optics_alignment_regression/train.py --lr 0.0001
```

---
**NOTE**
for using --binsize N (e.g --binsize 8), you have to run the make_processed.py with this set up first to get the data for training.
---

## 7. Evaluation

Script: `src/optics_alignment_regression/evaluate.py`. Output: `output/evaluation_results.csv`

- A normal evaluation for neural network process
- We save the setup and the loss to the output file.

### Usage
Here is 4 set up we've run
```bash
    PYTHONPATH="$PWD" python src/optics_alignment_regression/evaluate.py --lr 0.1
    PYTHONPATH="$PWD" python src/optics_alignment_regression/evaluate.py --lr 0.01
    PYTHONPATH="$PWD" python src/optics_alignment_regression/evaluate.py --lr 0.001
    PYTHONPATH="$PWD" python src/optics_alignment_regression/evaluate.py --lr 0.0001
```

---
**NOTE**
- For using --binsize N (e.g --binsize 8), you have to run the make_processed.py with this set up first to get the data for training.
- Then you have to run the train.py with the same setup first.
---