import Desaturation as ds
import cv2
import numpy as np
from PIL import Image
def median_cut_processing(image_path):
    try:
        original_img = Image.open(image_path)
        color_quantized = original_img.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
        original_img.show(title="Original")
        color_quantized.show(title="Median Cut (16 Colors)")
        color_quantized.save("output_median_cut_color.png")
        print("Images saved successfully.")

    except FileNotFoundError:
        print("Error: Could not find the file. Check your path.")

if __name__ == "__main__":
    path = "image.jpeg"
    median_cut_processing(path)