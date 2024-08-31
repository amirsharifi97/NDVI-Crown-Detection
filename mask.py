import os
import shutil
import tkinter as tk
from tkinter import filedialog
from tkinter import DoubleVar

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage import morphology, measure, segmentation
from skimage.segmentation import find_boundaries
import cv2
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import gaussian
from skimage.morphology import remove_small_holes, remove_small_objects

# Initialize the main window
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

root = ctk.CTk()
root.title("NDVI Tree Crown Detection")
root.geometry("700x500")

# Variables to store file path and threshold
file_path = None
ndvi_threshold = ctk.DoubleVar(value=0.5)
vegetation_mask = None  # Global variable for vegetation mask
rgb_image = None  # Global variable for RGB image

sigma_value = ctk.DoubleVar(value=1.0)  # Default value for sigma
h_value = ctk.DoubleVar(value=0.1)       # Default value for h

def select_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
    if file_path:
        file_label.configure(text=f"Selected file: {os.path.basename(file_path)}")

def process_image():
    global file_path, ndvi_threshold, vegetation_mask, rgb_image, final_mask
    
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

    def stretch_band(band):
        p2, p98 = np.percentile(band, (2, 98))
        band = np.clip(band, p2, p98)
        return (band - p2) / (p98 - p2) * 255

    def enforce_edge_connectivity(mask):
        mask[0, :] = mask[0, :] | mask[1, :]
        mask[-1, :] = mask[-1, :] | mask[-2, :]
        mask[:, 0] = mask[:, 0] | mask[:, 1]
        mask[:, -1] = mask[:, -1] | mask[:, -2]
        return mask
    def correct_isolated_black_pixels(final_mask, vegetation_mask):
    # Create a padded version of the mask to handle edge cases
        padded_final_mask = np.pad(final_mask, pad_width=1, mode='constant', constant_values=0)
        padded_vegetation_mask = np.pad(vegetation_mask, pad_width=1, mode='constant', constant_values=0)
        
        # Iterate over the original mask (without the padding)
        for i in range(1, final_mask.shape[0] + 1):
            for j in range(1, final_mask.shape[1] + 1):
                # Check if the current pixel is black in final_mask
                if padded_final_mask[i, j] == 0:
                    # Check if all surrounding pixels are black in final_mask
                    surrounding_pixels = padded_final_mask[i-1:i+2, j-1:j+2]
                    if np.all(surrounding_pixels == 0):
                        # Check if the corresponding pixel in vegetation_mask is white
                        if padded_vegetation_mask[i, j] == 1:
                            # Change the pixel value in final_mask to white
                            final_mask[i-1, j-1] = 255  # Unpad index to reflect original size

    R_vis = stretch_band(R).astype(np.uint16)
    G_vis = stretch_band(G).astype(np.uint16)
    B_vis = stretch_band(B).astype(np.uint16)
    N_vis = stretch_band(N).astype(np.uint16)

    # Stack the RGB bands into a single image array for visualization
    r_new = R_vis / 255.0
    n_new = N_vis / 255.0
    b_new = B_vis / 255.0
    g_new = G_vis / 255.0
    R_I = np.stack((R_vis, G_vis, B_vis), axis=-1)

    ndvi = (n_new - r_new) / (n_new + r_new + 1e-10)
    ############ Enhanced Vegetation Index ############
    # ndvi = 2.4*((n_new - r_new) / (n_new + r_new + 1))

    # Apply NDVI threshold
    vegetation_mask = ndvi > ndvi_threshold.get()

    # Morphological operations
    vegetation_mask = morphology.erosion(vegetation_mask, morphology.disk(2))
    vegetation_mask = morphology.dilation(vegetation_mask, morphology.disk(5))

    distance = distance_transform_edt(vegetation_mask)
    distance_smooth = gaussian(distance, sigma_value.get())
    local_maxi = morphology.h_maxima(distance_smooth, h_value.get())
    markers = measure.label(local_maxi)
    labels = watershed(-distance_smooth, markers, mask=vegetation_mask)

    original_with_boundaries = np.copy(R_I)
    boundary_color = np.array([255, 255, 0], dtype=np.uint8)

    boundary_mask = np.zeros_like(R_I[:, :, 0], dtype=bool)
    for label in np.unique(labels):
        if label == 0:
            continue

        crown_mask = labels == label
        crown_boundaries = measure.find_contours(crown_mask, 0.5)

        for contour in crown_boundaries:
            rr, cc = np.round(contour).astype(int).T
            rr = np.clip(rr, 0, boundary_mask.shape[0] - 1)
            cc = np.clip(cc, 0, boundary_mask.shape[1] - 1)
            boundary_mask[rr, cc] = True

    boundary_colored = np.zeros_like(original_with_boundaries)
    boundary_colored[boundary_mask] = boundary_color
    mask_from_boundary_colored = np.any(boundary_colored != 0, axis=-1)

    # Fill the regions inside the boundaries, including those touching the image edges
    filled_regions = binary_fill_holes(mask_from_boundary_colored)

    final_mask = np.zeros_like(mask_from_boundary_colored, dtype=np.uint8)
    final_mask[filled_regions] = 255
    final_mask[boundary_mask] = 0

    # Correct isolated black pixels
    correct_isolated_black_pixels(final_mask, vegetation_mask)

    original_with_boundaries = np.where(boundary_colored != 0, boundary_colored, original_with_boundaries)

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 4, 1)
    plt.title('NDVI')
    plt.imshow(ndvi, cmap='gray')
    plt.subplot(1, 4, 2)
    plt.title('Image')
    plt.imshow(R_I)
    plt.subplot(1, 4, 3)
    plt.title('Boundaries')
    plt.imshow(final_mask, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.title('Boundaries on Original Image')
    plt.imshow(original_with_boundaries)
    plt.show()

    boundary_count_label.configure(text=f"Total number of distinct tree crown boundaries: {len(np.unique(labels)) - 1}")

def save_image():
    global final_mask, file_path
    if final_mask is not None and file_path is not None:
        # Create the directories
        masks_dir = os.path.join(os.path.dirname(file_path), 'masks')
        masked_images_dir = os.path.join(os.path.dirname(file_path), 'masked_images')
        
        # Create directories if they don't exist
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)
        if not os.path.exists(masked_images_dir):
            os.makedirs(masked_images_dir)
        
        # Save the binary mask
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(masks_dir, f"{base_name}.png")
        cv2.imwrite(save_path, (final_mask).astype(np.uint8))
        print(f"Saved binary mask to {save_path}")
        
        # Move the original image to the masked_images directory
        new_file_path = os.path.join(masked_images_dir, os.path.basename(file_path))
        shutil.move(file_path, new_file_path)
        print(f"Moved original image to {new_file_path}")
        
        # Update the file path to the new location
        file_path = new_file_path

def delete_file():
    global file_path
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
        file_label.configure(text="No file selected")
        file_path = None  # Reset the file path variable
    else:
        print("No file selected or file doesn't exist")

def manual_rename():
    global file_path
    if file_path and os.path.exists(file_path):
        # Extract directory and filename
        directory, filename = os.path.split(file_path)
        
        # Create the "Manuals" directory if it doesn't exist
        manuals_dir = os.path.join(directory, "Manuals")
        if not os.path.exists(manuals_dir):
            os.makedirs(manuals_dir)
        
        # Prepend "MANUAL_" to the filename
        new_filename = "MANUAL_" + filename
        
        # Set the new file path in the "Manuals" directory
        new_file_path = os.path.join(manuals_dir, new_filename)
        
        # Move and rename the file
        shutil.move(file_path, new_file_path)
        print(f"Moved and renamed file to: {new_file_path}")
        
        # Update the file path and label
        file_path = new_file_path
        file_label.configure(text=f"Selected file: {os.path.basename(file_path)}")
    else:
        print("No file selected or file doesn't exist")

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

threshold_slider = ctk.CTkSlider(threshold_frame, from_=0, to=1, variable=ndvi_threshold ,  number_of_steps=100)
threshold_slider.pack(side=tk.LEFT, padx=5)

threshold_entry = ctk.CTkEntry(threshold_frame, textvariable=ndvi_threshold, width=50)
threshold_entry.pack(side=tk.LEFT, padx=5)

# Frame for sigma and h inputs (side by side)
parameters_frame = ctk.CTkFrame(root)
parameters_frame.pack(pady=15)

sigma_label = ctk.CTkLabel(parameters_frame, text="Sigma")
sigma_label.grid(row=0, column=0, padx=10)

sigma_entry = ctk.CTkEntry(parameters_frame, textvariable=sigma_value, width=50)
sigma_entry.grid(row=0, column=1, padx=10)

h_label = ctk.CTkLabel(parameters_frame, text="h")
h_label.grid(row=0, column=2, padx=10)

h_entry = ctk.CTkEntry(parameters_frame, textvariable=h_value, width=50)
h_entry.grid(row=0, column=3, padx=10)

# Show and Save buttons
button_frame = ctk.CTkFrame(root)
button_frame.pack(pady=20)

show_button = ctk.CTkButton(button_frame, text="Show", command=process_image, width=150)
show_button.grid(row=0, column=0, padx=10)

save_button = ctk.CTkButton(button_frame, text="Save Mask", command=save_image, width=150)
save_button.grid(row=0, column=1, padx=10)

# Manual button
manual_button = ctk.CTkButton(root, text="Manual", command=manual_rename, width=150)
manual_button.pack(pady=10)

# Delete button
delete_button = ctk.CTkButton(root, text="Delete File", command=delete_file, width=150, fg_color="red")
delete_button.pack(side=tk.BOTTOM, pady=20)

# Boundary count label
boundary_count_label = ctk.CTkLabel(root, text="")
boundary_count_label.pack(pady=10)

# Start the GUI loop
root.mainloop()
