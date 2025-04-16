import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cellpose import models, utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load Pre-trained Cellpose Model
model = models.Cellpose(model_type='cyto')  # For general cell detection
# Use model_type='nuclei' for nuclei detection if required

# Directory containing cell images
image_dir = "cells"  # Replace with your image folder path
output_dir = "cell_detection_results"
os.makedirs(output_dir, exist_ok=True)

# Parameters for Cellpose
channels = [0, 0]  # Grayscale input
diameter = 90    # Automatic diameter estimation; can specify manually (e.g., 30 for small cells)

# Kernel for dilation to make outlines thicker (3x3 or larger)
kernel = np.ones((3, 3), np.uint8)  # You can increase the size (e.g., (7,7)) for thicker lines

# Process each image in the directory
for image_file in os.listdir(image_dir):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
        image_path = os.path.join(image_dir, image_file)

        # Load the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Run Cellpose model
        masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels)

        # Debug: Inspect masks
        print(f"Processing {image_file}")
        print(f"Masks shape: {masks.shape}, Unique values: {np.unique(masks)}")

        # Check if any cells are detected
        if np.max(masks) == 0:
            print(f"No cells detected in {image_file}. Skipping.")
            continue

        # Convert masks to outlines
        outlines = utils.masks_to_outlines(masks)

        # Debug: Inspect outlines
        print(f"Outlines unique values: {np.unique(outlines)}")

        # Dilate the outlines to make them thicker
        thickened_outlines = cv2.dilate(outlines.astype(np.uint8), kernel, iterations=1)

        # Create an RGB overlay (3 channels)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Highlight the outlines in red (RGB: [255, 0, 0])
        overlay[thickened_outlines > 0] = [255, 0, 0]  # Red color for outlines

        # Save the overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title("Cell Detection")
        plt.axis("off")
        output_path = os.path.join(output_dir, f"detected_{image_file}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Processed and saved: {output_path}")

print("Cell detection completed.")
