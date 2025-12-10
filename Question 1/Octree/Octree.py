from PIL import Image
import os

FILE_SOURCE = "image.jpeg" 
def quantize_octree(image_path, k):
    try:
        img = Image.open(image_path).convert('RGB')
        
        print(f"Original Image Size: {img.size}")
        print(f"Quantizing to {k} colors using Fast Octree (Method 2)...")
        
        # method=2 enforces the Fast Octree algorithm in Pillow
        quantized_img = img.quantize(colors=k, method=2)
        
        return quantized_img
        
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Check if the hardcoded file exists before proceeding
    if not os.path.exists(FILE_SOURCE):
        print(f"Error: Could not find file at: {FILE_SOURCE}")
        print("Please update the 'FILE_SOURCE' variable in the code.")
    else:
        try:
            # Take K value input from the user
            k_input = input("Enter the number of colors (k) for quantization (e.g., 16, 64, 256): ")
            k_value = int(k_input)
            
            # Validate K
            if k_value < 1 or k_value > 256:
                print("Warning: k usually works best between 2 and 256.")

            # Run Quantization
            result_image = quantize_octree(FILE_SOURCE, k_value)
            if result_image:
                print("Quantization complete.")
                result_image.show()
                output_filename = f"octree_quantized_k{k_value}.png"
                result_image.save(output_filename)
                print(f"Image saved as {output_filename}")

        except ValueError:
            print("Invalid input. Please enter an integer for k.")