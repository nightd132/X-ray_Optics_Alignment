import torch
from src.optics_alignment_regression.data_preprocessing import preprocess_image, load_fabio_image
from src.optics_alignment_regression.utils import load_checkpoint
import numpy as np
class OpticsAlignmentPredictor():
    def __init__(self, ckp_path, device="cpu"):
        self.device = torch.device(device)
        self.model, self.cfg = load_checkpoint(ckp_path, self.device)
    def remove_bars(self, img, invalid_fnc=lambda a: a < 0):
        valid = ~invalid_fnc(img)
        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        return img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    
    def predict(self, image_path):
        raw_image = load_fabio_image(image_path)
        preprocessed_image = self.remove_bars(preprocess_image(raw_image, self.cfg["binsize"]))
        image_tensor = torch.from_numpy(
            preprocessed_image.astype("float32")
        )

        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        # Move tensor to the same device as the model
        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        predicted_offset_x = predictions[0, 0].item()
        predicted_offset_z = predictions[0, 1].item()
        return {
            "raw_image": raw_image,
            "preprocessed_image": preprocessed_image,
            "predicted_offset_x": predicted_offset_x,
            "predicted_offset_z": predicted_offset_z
        }
