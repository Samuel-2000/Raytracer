import numpy as np
import cv2

class Denoiser:
    """Denoising algorithms"""
    
    def __init__(self):
        self.available_methods = ['bilateral', 'nlmeans', 'gaussian', 'median', 'neural']
    
    def denoise(self, image: np.ndarray, method: str = 'bilateral', **kwargs) -> np.ndarray:
        """Apply denoising to image"""
        # Vstup je v rozsahu [0,1], prevod na uint8
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        if method == 'bilateral':
            return self._bilateral_denoise(image_uint8, **kwargs)
        elif method == 'nlmeans':
            return self._nlmeans_denoise(image_uint8, **kwargs)
        elif method == 'gaussian':
            return self._gaussian_denoise(image_uint8, **kwargs)
        elif method == 'median':
            return self._median_denoise(image_uint8, **kwargs)
        elif method == 'neural':
            return self._neural_denoise(image_uint8, **kwargs)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def _bilateral_denoise(self, image: np.ndarray, d: int = 9, 
                          sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised.astype(np.float32) / 255.0
    
    def _nlmeans_denoise(self, image: np.ndarray, h: float = 10,
                        template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window_size, search_window_size
        )
        return denoised.astype(np.float32) / 255.0
    
    def _gaussian_denoise(self, image: np.ndarray, kernel_size: int = 5,
                         sigma: float = 1.0) -> np.ndarray:
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return denoised.astype(np.float32) / 255.0
    
    def _median_denoise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        denoised = cv2.medianBlur(image, kernel_size)
        return denoised.astype(np.float32) / 255.0
    
    def _neural_denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Neural network denoising.
        TODO: Implement actual neural network inference (e.g., OpenCV DNN).
        """
        # Placeholder: return original image
        print("Neural denoising not implemented yet, returning original image.")
        return image.astype(np.float32) / 255.0