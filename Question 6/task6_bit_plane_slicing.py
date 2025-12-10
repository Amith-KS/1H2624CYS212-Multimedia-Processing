import cv2
import numpy as np
import os

def run_bit_plane_slicing(image_path):
    print(f"\n--- Processing: {image_path} ---")
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # Load image in grayscale
    img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    base_name=os.path.splitext(image_path)[0]
    output_dir=f"{base_name}_planes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Compute & Save 8 Bit-planes
    print(f"Saving bit-planes to: {output_dir}/")
    for i in range(8):
        mask=1<<i
        plane=cv2.bitwise_and(img,mask)
        # Scale for visualization
        visual_plane=(plane>0).astype(np.uint8)*255
        cv2.imwrite(f"{output_dir}/plane_{i}.jpg",visual_plane)
        print(f" -> Saved Bit-plane {i}")

    # 2. Union of 3 Lowest Bit-planes
    print("Computing Union of Lowest 3 Planes...")
    mask_lowest_3=7
    union_lowest_3=cv2.bitwise_and(img,mask_lowest_3)
    cv2.imwrite(f"{base_name}_union_lowest_3_visual.jpg",union_lowest_3*30)

    # 3. Difference the Union from Original
    print("Differencing Union from Original...")
    diff_image=cv2.subtract(img,union_lowest_3)
    cv2.imwrite(f"{base_name}_final_difference.jpg",diff_image)
    
    print(f" -> Saved: {base_name}_union_lowest_3_visual.jpg")
    print(f" -> Saved: {base_name}_final_difference.jpg")

if __name__=="__main__":
    images_to_process=[r"C:\Users\AMITH\D\COLLEGE FILES\MULTIMEDIA\1H2624CYS212-Multimedia-Processing\Question 6\low_light.jpg",
                       r"C:\Users\AMITH\D\COLLEGE FILES\MULTIMEDIA\1H2624CYS212-Multimedia-Processing\Question 6\bright_light.jpg"]
    
    for img_name in images_to_process:
        run_bit_plane_slicing(img_name)