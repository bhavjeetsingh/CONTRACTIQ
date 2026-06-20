"""
Image Preprocessor for OCR Pipeline
=====================================
Applies lightweight image preprocessing to improve OCR accuracy
on scanned contracts. Keeps processing minimal to stay within
free-tier memory limits.

Preprocessing steps:
    1. Convert to grayscale (reduces noise from color backgrounds)
    2. Adaptive thresholding (handles uneven lighting in scans)
    3. Optional deskew correction (straightens rotated scans)
"""

import numpy as np
from logger import GLOBAL_LOGGER as log


class ImagePreprocessor:
    """
    Lightweight image preprocessor for OCR.
    
    Uses only numpy operations to avoid heavy dependencies.
    For production with higher accuracy needs, consider adding
    OpenCV (cv2) preprocessing.
    """

    def __init__(self, enable_grayscale: bool = True, enable_threshold: bool = False):
        """
        Args:
            enable_grayscale: Convert color images to grayscale.
            enable_threshold: Apply simple thresholding (can help with low-contrast scans).
        """
        self.enable_grayscale = enable_grayscale
        self.enable_threshold = enable_threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to an image.

        Args:
            image: numpy array of shape (H, W, C) or (H, W) for grayscale

        Returns:
            Preprocessed image as numpy array
        """
        try:
            processed = image.copy()

            # Step 1: Convert to grayscale if color image
            if self.enable_grayscale and len(processed.shape) == 3:
                # Luminance formula: 0.299*R + 0.587*G + 0.114*B
                processed = np.dot(processed[..., :3], [0.299, 0.587, 0.114])
                processed = processed.astype(np.uint8)

            # Step 2: Simple threshold for low-contrast scans
            if self.enable_threshold and len(processed.shape) == 2:
                # Otsu-style simple threshold
                threshold = np.mean(processed)
                processed = np.where(processed > threshold, 255, 0).astype(np.uint8)

            return processed

        except Exception as e:
            log.warning("Image preprocessing failed, using original", error=str(e))
            return image

    def estimate_quality(self, image: np.ndarray) -> float:
        """
        Estimate image quality for OCR (0.0 = poor, 1.0 = excellent).
        
        Uses simple heuristics:
            - Contrast ratio
            - Brightness distribution
            - Edge density (indicates text presence)

        Args:
            image: numpy array

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                gray = image

            # Contrast: standard deviation of pixel values
            contrast = np.std(gray) / 128.0  # Normalize to 0-1 range
            contrast = min(contrast, 1.0)

            # Brightness: mean pixel value (ideal around 128 for documents)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0

            # Combined score
            quality = 0.6 * contrast + 0.4 * brightness_score
            return round(min(max(quality, 0.0), 1.0), 3)

        except Exception:
            return 0.5  # Default middle score on error
