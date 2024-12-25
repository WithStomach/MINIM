import numpy as np
import pydicom
from typing import Dict, Union, Tuple
import nibabel as nib
from scipy import ndimage
from PIL import Image

class MedicalImageProcessor:
    def __init__(self, config: Dict):
        self.config = config
        image_size = config["synthetic_system"]["image_size"][0]
        self.target_size = (image_size, image_size)
        
    def load_medical_image(self, path: str) -> np.ndarray:
        if isinstance(path, (Image.Image, np.ndarray)):
            if isinstance(path, Image.Image):
                image = np.array(path)
            else:
                image = path
            return self._preprocess_image(image)
            
        if path.endswith(('.dcm', '.DCM')):
            return self._load_dicom(path)
        elif path.endswith(('.nii', '.nii.gz')):
            return self._load_nifti(path)
        else:
            return self._load_standard_image(path)
    
    def _load_dicom(self, path: str) -> np.ndarray:
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(float)
        
        if hasattr(dcm, 'Modality') and dcm.Modality == 'CT':
            center = self.config["synthetic_system"]["medical_settings"]["preprocessing"]["window_center"]
            width = self.config["synthetic_system"]["medical_settings"]["preprocessing"]["window_width"]
            image = self._apply_window_level(image, center, width)
            
        return self._preprocess_image(image)
    
    def _load_nifti(self, path: str) -> np.ndarray:
        nifti = nib.load(path)
        image = nifti.get_fdata()
        
        if len(image.shape) == 3:
            image = image[:, :, image.shape[2]//2]
            
        return self._preprocess_image(image)
    
    def _load_standard_image(self, path: str) -> np.ndarray:
        with Image.open(path).convert('L') as img:
            image = np.array(img)
        return self._preprocess_image(image)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
            
        image_pil = image_pil.resize(self.target_size, Image.Resampling.LANCZOS)
        image = np.array(image_pil)
        
        if self.config["synthetic_system"]["medical_settings"]["preprocessing"]["normalize"]:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
        return image
    
    def _apply_window_level(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        min_value = center - width//2
        max_value = center + width//2
        image = np.clip(image, min_value, max_value)
        return (image - min_value) / (max_value - min_value)
    
    def compute_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        metrics = {}
        metrics['snr'] = np.mean(image) / np.std(image)
        metrics['contrast'] = np.max(image) - np.min(image)
        gradient = ndimage.sobel(image)
        metrics['artifact_score'] = np.mean(np.abs(gradient))
        return metrics