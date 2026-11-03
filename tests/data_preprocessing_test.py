from src.optics_alignment_regression.data_preprocessing import (
    load_fabio_image,
    preprocess_image,
    save_preprocessed_image_as_h5,
    find_brightest_point,
    find_optimal_image,
    load_log_txt_file
)

def test_load_image():
    # Test loading a valid image
    image_path = "data/Mainz_2022-10_AXO-Optics-Scan-forML/Mainz_20220912_03_AXO-Optics-Scan-forML_245W_Al-tape_noFlTu_SourceDet-130cm_detPos2/image_0000001.cbf"
    image = load_fabio_image(image_path)
    assert image is not None, "Failed to load a valid image."

    # Test loading an invalid image
    invalid_image_path = "tests/test_data/invalid_image.cbf"
    try:
        load_fabio_image(invalid_image_path)
        assert False, "Expected an exception for an invalid image path."
    except FileNotFoundError:
        pass  # Expected exception

def test_preprocess_image():
    # Test preprocessing a valid image
    # Image shape: (1043, 981)
    # Preprocessed image shape: (260, 245) for bin_size=4
    image_path = "data/Mainz_2022-10_AXO-Optics-Scan-forML/Mainz_20220912_03_AXO-Optics-Scan-forML_245W_Al-tape_noFlTu_SourceDet-130cm_detPos2/image_0000001.cbf"
    image = load_fabio_image(image_path)
    bin_size = 4
    preprocessed_image = preprocess_image(image, bin_size)
    assert preprocessed_image is not None, "Failed to preprocess a valid image."
    assert preprocessed_image.ndim == 2, "Preprocessed image should be 2D."
    assert preprocessed_image.shape[0] == image.shape[0] // bin_size, "Unexpected height of preprocessed image."
    assert preprocessed_image.shape[1] == image.shape[1] // bin_size, "Unexpected width of preprocessed image."

def test_load_log_txt_file():
    # Test loading a valid log file
    # hexapod x = 0.312300
    # hexapod y = -0.001000
    # hexapod z = -1.512900
    log_file_path = "data/Mainz_2022-10_AXO-Optics-Scan-forML/Mainz_20220912_03_AXO-Optics-Scan-forML_245W_Al-tape_noFlTu_SourceDet-130cm_detPos2/log_0000001.txt"
    hexapod_x, hexapod_z = load_log_txt_file(log_file_path)
    assert hexapod_x is not None and hexapod_z is not None, "Failed to extract offsets from a valid log file."
    assert isinstance(hexapod_x, float) and isinstance(hexapod_z, float), "Offsets should be of type float."
    assert hexapod_x == 0.312300, "Unexpected x offset value."
    assert hexapod_z == -1.512900, "Unexpected z offset value."

    # Test loading an invalid log file
    invalid_log_file_path = "tests/test_data/invalid_log.txt"
    try:
        load_log_txt_file(invalid_log_file_path)
        assert False, "Expected an exception for an invalid log file path."
    except FileNotFoundError:
        pass  # Expected exception

