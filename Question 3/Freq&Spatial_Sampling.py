import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_dummy_image(size=512):
    """Creates a synthetic checkerboard image if no file is found."""
    img = np.zeros((size, size), dtype=np.uint8)
    block_size = size // 8
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i:i+block_size, j:j+block_size] = 255
    return img

def spatial_sampling(image, factor):
    """
    Performs Spatial Sampling (Downsampling).
    """
    # Take every k-th pixel
    sampled_image = image[::factor, ::factor]
    return sampled_image

def frequency_sampling(image, keep_fraction):
    """
    Performs Frequency Sampling (Bandwidth Limitation).
    """
    # 1. FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 2. Mask
    mask = np.zeros((rows, cols), np.uint8)
    r_rad = int((rows * keep_fraction) / 2)
    c_rad = int((cols * keep_fraction) / 2)
    
    # Keep center
    mask[crow-r_rad:crow+r_rad, ccol-c_rad:ccol+c_rad] = 1
    
    # 3. Inverse FFT
    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    
    return np.abs(img_back)

# --- Main Execution Block ---

# 1. Load Image
filename = 'image.jpeg'  # Ensure this matches your file name (e.g., .jpeg vs .jpg)
img = cv2.imread(filename, 0) # 0 loads as grayscale

if img is None:
    print(f"Warning: '{filename}' not found. Using synthetic image.")
    img = create_dummy_image()

# Get original dimensions to fix the axis scale later
original_h, original_w = img.shape

# 2. Define sampling factors
factors = [2, 4, 8,16] 

# 3. Plotting Setup
plt.figure(figsize=(12, 8))

# --- Show Original ---
plt.subplot(3, len(factors) + 1, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Original\n{img.shape}')
# Set axis limits to original size
plt.xlim(0, original_w)
plt.ylim(original_h, 0)
plt.axis('on')

# --- Loop through factors ---
for i, f in enumerate(factors):
    # -----------------------------
    # A. Spatial Sampling
    # -----------------------------
    s_res = spatial_sampling(img, f)
    
    # SAVE SPATIAL IMAGE
    spatial_filename = f"spatial_sampling_1_{f}.jpg"
    cv2.imwrite(spatial_filename, s_res)
    print(f"Saved: {spatial_filename}")

    # Plot Spatial
    sp_plot_idx = i + 2 + (len(factors) + 1)
    ax_sp = plt.subplot(3, len(factors) + 1, sp_plot_idx)
    ax_sp.imshow(s_res, cmap='gray')
    ax_sp.set_title(f'Spatial 1/{f}\n{s_res.shape}')
    ax_sp.set_xlim(0, original_w)
    ax_sp.set_ylim(original_h, 0) 
    
    # -----------------------------
    # B. Frequency Sampling
    # -----------------------------
    freq_res = frequency_sampling(img, 1/f)
    
    # SAVE FREQUENCY IMAGE
    # freq_res is floating point. Normalize/Clip to 0-255 and convert to uint8 for saving.
    # This prevents saving a black or corrupted image.
    freq_res_save = np.clip(freq_res, 0, 255).astype(np.uint8)
    
    freq_filename = f"frequency_sampling_1_{f}.jpg"
    cv2.imwrite(freq_filename, freq_res_save)
    print(f"Saved: {freq_filename}")

    # Plot Frequency
    fr_plot_idx = i + 2 + 2 * (len(factors) + 1)
    ax_fr = plt.subplot(3, len(factors) + 1, fr_plot_idx)
    ax_fr.imshow(freq_res, cmap='gray')
    ax_fr.set_title(f'Frequency 1/{f}\n(Blurry)')
    ax_fr.set_xlim(0, original_w)
    ax_fr.set_ylim(original_h, 0)
    ax_fr.axis('off')

plt.tight_layout()
plt.show()