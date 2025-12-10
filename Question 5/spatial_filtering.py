import cv2
import numpy as np
import os

def run_task_5(filename):
    print(f"--- Processing: {filename} ---")
    if not os.path.exists(filename):
        print("File not found")
        return

    img=cv2.imread(filename)

    # Box Filters
    print("Applying Box Filters...")
    for size in [5,20]:
        # Normalized
        box_norm=cv2.blur(img,(size,size))
        cv2.imwrite(f"box_{size}x{size}_normalized.jpg",box_norm)
        
        # Non-Normalized
        kernel_raw=np.ones((size,size),dtype=np.float32)
        box_non_norm=cv2.filter2D(img,-1,kernel_raw)
        cv2.imwrite(f"box_{size}x{size}_non_normalized.jpg",box_non_norm)
        print(f"Saved {size}x{size} filters")

    # Compute Sigma & Gaussian
    print("Computing Sigma & Gaussian...")
    box_k=20
    # Formula: sigma = 0.3*((k-1)*0.5 - 1) + 0.8
    computed_sigma=0.3*((box_k-1)*0.5-1)+0.8
    print(f"Computed Sigma: {computed_sigma:.2f}")
    
    # Compute filter size from sigma
    gauss_k=int(round(2*np.pi*computed_sigma))
    if gauss_k%2==0:
        gauss_k+=1
    print(f"Filter Size: {gauss_k}")

    # Create 1D Kernel
    center=gauss_k//2
    x_vals=np.arange(-center,center+1)
    gauss_1d=np.exp(-(x_vals**2)/(2*computed_sigma**2))
    
    # Normalized & Non-Normalized Kernels
    gauss_norm=gauss_1d/np.sum(gauss_1d)
    gauss_raw=gauss_1d

    # Apply Separable
    sep_norm=cv2.sepFilter2D(img,-1,gauss_norm,gauss_norm)
    cv2.imwrite(f"gaussian_sigma{computed_sigma:.1f}_norm.jpg",sep_norm)
    
    sep_raw=cv2.sepFilter2D(img,-1,gauss_raw,gauss_raw)
    cv2.imwrite(f"gaussian_sigma{computed_sigma:.1f}_non_norm.jpg",sep_raw)
    
    print("Done")

if __name__=="__main__":
    file_name=r"C:\Users\AMITH\D\COLLEGE FILES\MULTIMEDIA\1H2624CYS212-Multimedia-Processing\Question 5\Torgya - Arunachal Festival.jpg"
    run_task_5(file_name)