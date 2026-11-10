from src.optics_alignment_regression.dataset import (
    load_meta_data, 
    load_h5_dataset, 
    load_split_h5
)

import numpy as np

def test_load_meta_data():
    path = "data/Mainz_2022-10_AXO-Optics-Scan-forML/Mainz_20220130_19_AXO-Optics-Scan-forML_250W_Al-tape_noFlTu_SourceDet-236cm_focus/log_0000001.txt"

    meta_data = load_meta_data(path)
    assert len(meta_data) == 37
    assert meta_data["hexapod_x"] == 1.084200
    assert meta_data["hexapod_z"] == -0.965400