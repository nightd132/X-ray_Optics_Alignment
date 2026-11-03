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


## 2. Dataset Preprocessing

Script: `src/optics_alignment_regression/data_preprocess.py`. Output: `data/preprocessed_data.h5`

### Labels
- For each scan folder (one detector distance + camera position), the **optimal image** is the raw image with the highest intensity point.
- The intensity is found on the **raw** image, before binning, so labels do not depend on `bin_size`.
- The hexapod (x, z) of that image, read from `log_<id>.txt`, is the reference.
- Each image's target is `offset = hexapod_pos(image) - hexapod_pos(best)`.

Units: same as the hexapod log. Sign: positive = image is displaced in +x / +z from the optimum. The network predicts this offset; to correct, move the hexapod by `-offset`.

### Preprocessing
- Binning: 4x4 mean, edges cropped to a multiple of 4.
- Invalid pixels set to 0 before binning.
- Normalisation is stored (all pixels in range [0, 1]).

### HDF5 layout
data_point_<N>/
  image     float32, (H/12, W/12)
  x_offset  float64, scalar
  z_offset  float64, scalar

### Usage
```bash
    PYTHONPATH="$PWD" python src/optics_alignment_regression/data_preprocessing.py
```

## 3. Data Splitting

Script: `src/optics_alignment_regression/make_split.py`. Output: `data/split_data.h5`

- Split the preprocessed_dataset into train and test set
- Store as split_data.h5

### Split file layout
train/
  image     float32, (N_train, H/12, W/12)
  targets   float32, (N_train, 2)   # [x_offset, z_offset]
test/
  image     float32, (N_test, H/12, W/12)
  targets   float32, (N_test, 2)

### Usage
```bash
    PYTHONPATH="$PWD" python src/optics_alignment_regression/make_split.py
```