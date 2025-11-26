import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

try:
    image = plt.imread("image.jpeg")
except FileNotFoundError:
    print("Image 'test.jpg' not found. Please ensure the file exists.")

# 2. Preprocess the image
X = image.reshape(-1, 3)
n_clusters=int(input("Enter k value: "))
# 3. Define and Train the K-Means model
kmeans = KMeans(n_clusters, n_init=10, random_state=42)
kmeans.fit(X)

# 4. Reconstruct the Segmented Image
segmented_data = kmeans.cluster_centers_[kmeans.labels_]

# Reshape the long list of colors back into the original image dimensions
segmented_image = segmented_data.reshape(image.shape)

# 5. Display the Results
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Segmented Image
plt.subplot(1, 2, 2)
plt.imsave(f"segmented_image_k{n_clusters}.png", segmented_image / 255.0)
plt.imshow(segmented_image/255.0)
plt.title(f"Segmented Image ({n_clusters} Colors)")
plt.axis('off')

plt.show()