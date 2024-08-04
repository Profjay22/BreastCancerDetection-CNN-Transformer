import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Define directories
normal_dir = Path("/data/ja235/camelyon16_project/data/full_all_patches/normal")
tumor_dir = Path("/data/ja235/camelyon16_project/data/full_all_patches/tumor")

# Function to extract slide ID, x, y, and patch type from filename
def extract_info(filename):
    parts = filename.split('-')
    slide_id = parts[0]
    x_coord = parts[1]
    y_coord = parts[2]
    return slide_id, x_coord, y_coord

# Function to process a directory and return a DataFrame
def process_directory(directory):
    data = []
    for root, _, files in tqdm(os.walk(directory), desc=f"Processing {directory}"):
        for file in sorted(files):  # Ensure files are processed in order
            if file.endswith(".png"):
                slide_id, x_coord, y_coord = extract_info(file)
                data.append((os.path.join(root, file), slide_id, x_coord, y_coord))
    return pd.DataFrame(data, columns=["patch_path", "slide_id", "x_coord", "y_coord"])

print("Starting to process directories...")

# Process both directories
normal_df = process_directory(normal_dir)
tumor_df = process_directory(tumor_dir)

print("Directories processed. Combining data...")

# Combine the DataFrames
all_patches_df = pd.concat([normal_df, tumor_df], ignore_index=True)

# Save the DataFrame to CSV
output_csv_path = "/data/ja235/camelyon16_project/fullpatch_positions.csv"
all_patches_df.to_csv(output_csv_path, index=False)

print(f"Patch positions and coordinates saved to {output_csv_path}.")
