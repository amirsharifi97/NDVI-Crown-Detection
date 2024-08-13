import os
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage import morphology, measure
from skimage.segmentation import find_boundaries
import cv2

# Initialize the main window
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

root = ctk.CTk()
root.title("NDVI Tree Crown Detection")
root.geometry("600x450")

# Variables to store file path and threshold
file_path = None
ndvi_threshold = ctk.DoubleVar(value=0.5)
vegetation_mask = None  # Global variable for vegetation mask
rgb_image = None  # Global variable for RGB image

def select_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
    if file_path:
        file_label.configure(text=f"Selected file: {os.path.basename(file_path)}")

def process_image():
    global file_path, ndvi_threshold, vegetation_mask, rgb_image
    
    if file_path is None:
        return
    
    # Read the TIFF file
    image = tiff.imread(file_path)

    # Extract the RGB and NIR bands
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    N = image[:, :, 3]

    # Normalize bands to 8-bit if necessary
    if N.dtype != np.uint8:
        N = 255 * (N - N.min()) / (N.max() - N.min())
        N = N.astype(np.uint8)
    
    rgb_image = np.stack((R, G, B), axis=-1)
    if rgb_image.dtype != np.uint8:
        image_min, image_max = rgb_image.min(), rgb_image.max()
        rgb_image = 255 * (rgb_image - image_min) / (image_max - image_min)
        rgb_image = rgb_image.astype(np.uint16)

    rgbn_image = np.dstack((rgb_image, N))

    r_new = rgbn_image[:, :, 0] / 255.0 
    g_new = rgbn_image[:, :, 1] / 255.0
    b_new = rgbn_image[:, :, 2] / 255.0
    n_new = rgbn_image[:, :, 3] / 255.0

    ndvi = (n_new - r_new) / (n_new + r_new + 1e-10)

    # Apply NDVI threshold
    vegetation_mask = ndvi > ndvi_threshold.get()

    # Morphological operations
    vegetation_mask = morphology.erosion(vegetation_mask, morphology.disk(2))
    vegetation_mask = morphology.dilation(vegetation_mask, morphology.disk(5))

    # Label connected regions
    labeled_tree_crowns = measure.label(vegetation_mask, connectivity=2)

    # Copy RGB image for overlaying boundaries
    original_with_boundaries = np.copy(rgb_image)

    # Overlay boundaries
    boundary_count = 0
    for label in np.unique(labeled_tree_crowns):
        if label == 0:
            continue
        crown_mask = labeled_tree_crowns == label
        crown_boundaries = find_boundaries(crown_mask, mode='outer')
        if np.any(crown_boundaries):
            boundary_count += 1
        thick_boundaries = morphology.dilation(crown_boundaries, morphology.disk(2))
        color = np.array([255, 255, 0], dtype=np.uint16)
        original_with_boundaries[thick_boundaries] = color

    # Display the images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('NDVI')
    plt.imshow(ndvi, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Image')
    plt.imshow(rgb_image)
    plt.subplot(1, 3, 3)
    plt.title('Boundaries on Original Image')
    plt.imshow(original_with_boundaries)
    plt.show()

    # Update the boundary count label
    boundary_count_label.configure(text=f"Total number of distinct tree crown boundaries: {boundary_count}")

def save_image():
    global vegetation_mask, file_path
    if vegetation_mask is not None and file_path is not None:
        # Create a save path by replacing the input file extension with .png
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_dir = os.path.join(os.path.dirname(file_path), 'masks')
        
        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f"{base_name}.png")
        cv2.imwrite(save_path, (vegetation_mask * 255).astype(np.uint8))
        print(f"Saved binary mask to {save_path}")

# Title
title_label = ctk.CTkLabel(root, text="NDVI Tree Crown Detection", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=20)

# File selection button and label
frame = ctk.CTkFrame(root)
frame.pack(pady=20, padx=20)

select_button = ctk.CTkButton(frame, text="Select TIFF Image", command=select_file, width=200)
select_button.grid(row=0, column=0, pady=5)

file_label = ctk.CTkLabel(frame, text="No file selected", width=40)
file_label.grid(row=0, column=1, pady=5, padx=10)

# NDVI threshold slider
threshold_frame = ctk.CTkFrame(root)
threshold_frame.pack(pady=10)

threshold_label = ctk.CTkLabel(threshold_frame, text="NDVI Threshold")
threshold_label.pack(side=tk.LEFT, padx=5)

threshold_slider = ctk.CTkSlider(threshold_frame, from_=0, to=1, variable=ndvi_threshold)
threshold_slider.pack(side=tk.LEFT, padx=5)

threshold_entry = ctk.CTkEntry(threshold_frame, textvariable=ndvi_threshold, width=50)
threshold_entry.pack(side=tk.LEFT, padx=5)

# Show and Save buttons
button_frame = ctk.CTkFrame(root)
button_frame.pack(pady=20)

show_button = ctk.CTkButton(button_frame, text="Show", command=process_image, width=150)
show_button.grid(row=0, column=0, padx=10)

save_button = ctk.CTkButton(button_frame, text="Save Mask", command=save_image, width=150)
save_button.grid(row=0, column=1, padx=10)

# Boundary count label
boundary_count_label = ctk.CTkLabel(root, text="")
boundary_count_label.pack(pady=10)

# Start the GUI loop
root.mainloop()
