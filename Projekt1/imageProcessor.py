from PIL import Image
import numpy as np


class ImageProcessor:
    def __init__(self, image_path):
        img_pil = Image.open(image_path).convert("RGB")
        self.pixels = np.array(img_pil, dtype=np.float32)
        self.original_pixels = self.pixels.copy() 
        self.height, self.width = self.pixels.shape[:2]

    def reset(self):
        self.pixels = self.original_pixels.copy()

    def get_image(self):
        img_uint8 = np.clip(self.pixels, 0, 255).astype(np.uint8)
        return Image.fromarray(img_uint8)

    # ----- pixel operations -----
    def grayscaleLum(self):
        gray = np.dot(self.pixels[..., :3], [0.299, 0.587, 0.114])
        self.pixels = np.stack((gray, gray, gray), axis=-1)

    def negative(self):
        self.pixels = 255 - self.pixels

    def brightness(self, value):
        self.pixels = np.clip(self.pixels + value, 0, 255)

    def contrast(self, factor):
        j_max = np.max(self.pixels)

        if j_max == 0:
            return
        self.pixels = 255.0 * ((self.pixels / j_max) ** factor)

    def binarize(self, threshold):
        self.grayscaleLum()
        self.pixels = np.where(self.pixels > threshold, 255, 0).astype(np.float32)

    # ----- filters -----
    def applyFilter(self, kernel):
        kernel = np.array(kernel, dtype=np.float32)
        kernel_sum = np.sum(kernel)
        if kernel_sum != 0:
            kernel /= kernel_sum 

        padded = np.pad(self.pixels, ((1, 1), (1, 1), (0, 0)), mode='edge')
        output = np.zeros_like(self.pixels)
        
        for i in range(3):
            for j in range(3):
                output += padded[i:i+self.height, j:j+self.width, :] * kernel[i, j]
                
        self.pixels = np.clip(output, 0, 255)

    # ----- edge detection -----
    def roberts(self):
        self.grayscaleLum()
        p1 = self.pixels[:-1, :-1, 0]
        p2 = self.pixels[:-1, 1:, 0]
        p3 = self.pixels[1:, :-1, 0]
        p4 = self.pixels[1:, 1:, 0]

        val = np.abs(p1 - p4) + np.abs(p2 - p3)
        
        output = np.zeros_like(self.pixels)
        output[:-1, :-1, 0] = val
        output[:-1, :-1, 1] = val
        output[:-1, :-1, 2] = val
        self.pixels = np.clip(output, 0, 255)


    def sobel(self):
        self.grayscaleLum()
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        
        padded = np.pad(self.pixels[..., 0], ((1, 1), (1, 1)), mode='edge')
        gx = np.zeros((self.height, self.width), dtype=np.float32)
        gy = np.zeros((self.height, self.width), dtype=np.float32)
        
        for i in range(3):
            for j in range(3):
                gx += padded[i:i+self.height, j:j+self.width] * kx[i, j]
                gy += padded[i:i+self.height, j:j+self.width] * ky[i, j]
                
        magnitude = np.sqrt(gx**2 + gy**2)
        self.pixels = np.stack((magnitude, magnitude, magnitude), axis=-1)
        self.pixels = np.clip(self.pixels, 0, 255)

    def get_histograms(self):
        is_gray = np.array_equal(self.pixels[..., 0], self.pixels[..., 1]) and \
                  np.array_equal(self.pixels[..., 1], self.pixels[..., 2])
        return is_gray, self.pixels

    def get_projections(self):
        lum = 0.299 * self.pixels[..., 0] + 0.587 * self.pixels[..., 1] + 0.114 * self.pixels[..., 2]

        inverted_lum = 255 - lum

        proj_h = np.sum(inverted_lum, axis=0)
        proj_v = np.sum(inverted_lum, axis=1)
        return proj_h, proj_v
