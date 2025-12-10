import cv2
import numpy as np

def desaturation_grayscale(img_bgr):
    img_float = img_bgr.astype(np.float32)
    max_val = np.max(img_float, axis=2)
    min_val = np.min(img_float, axis=2)
    grayscale_float = (max_val + min_val) / 2
    grayscale_img = grayscale_float.astype(np.uint8)
    return grayscale_img

image_bgr = cv2.imread(r"C:\Users\AMITH\D\COLLEGE FILES\MULTIMEDIA\image.jpeg")
gray_desat = desaturation_grayscale(image_bgr)
# cv2.imshow('Processed Grayscale', gray_desat)
cv2.imwrite('output_image.jpg', gray_desat)
cv2.waitKey(0)