import os
import glob
from PIL import Image
import numpy as np
import psutil
from scipy import ndimage as nd
from skimage import measure


def get_file_paths(directory, extensions=('*.tif', '*.png', '*.jpeg', '*.jpg')):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    log_message(f"Found {len(paths)} files in directory: {directory}")
    return paths

def create_mask_from_binary(mask_path):
    log_message(f"Creating mask from binary file: {mask_path}")
    mask = Image.open(mask_path)
    mask = np.array(mask.convert('L'))  # Convert to grayscale
    mask = (mask > 0).astype(np.uint8) * 255
    return mask

def save_patch(patch, directory, slide_id, prefix, index, is_positive):
    slide_dir = os.path.join(directory, slide_id)
    if not os.path.exists(slide_dir):
        os.makedirs(slide_dir)
    subdir = 'positive' if is_positive else 'negative'
    full_dir = os.path.join(slide_dir, subdir)
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    patch_path = os.path.join(full_dir, f"{prefix}{index}.png")
    patch.save(patch_path)
    log_message(f"Saved patch {index} for slide {slide_id} in directory: {full_dir}")

def log_message(message):
    print(message)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    log_message(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def compute_evaluation_mask(mask_path, resolution, level):
    mask = Image.open(mask_path)
    mask = np.array(mask.convert('L'))
    distance = nd.distance_transform_edt(255 - mask)
    Threshold = 75 / (resolution * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)
    return evaluation_mask
