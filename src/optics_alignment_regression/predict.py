import torch
from src.optics_alignment_regression.data_preprocessing import preprocess_image, load_fabio_image
from src.optics_alignment_regression.utils import load_checkpoint

class OpticsAlignmentPredictor():
    def __init__(self, ckp_path, device="cpu"):
        self.device = torch.device(device)
        self.model = load_checkpoint(ckp_path, self.device)

    def predict(self, image_path):
        raw_image = load_fabio_image(image_path)
        preprocessed_image = preprocess_image(raw_image)
        predicted_offset_x, predicted_offset_z = self.model(preprocessed_image)
        return {
            "raw_image": raw_image,
            "preprocessed_image": preprocessed_image,
            "predicted_offset_x": predicted_offset_x,
            "predicted_offset_z": predicted_offset_z
        }
