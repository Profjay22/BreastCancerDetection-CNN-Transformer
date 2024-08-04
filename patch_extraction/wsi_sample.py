# This code is adapted from https://wsipipe.readthedocs.io/en/latest/tutorial.html

from wsipipe.datasets import camelyon16
from wsipipe.load.datasets.camelyon16 import Camelyon16Loader
from wsipipe.preprocess.tissue_detection import TissueDetectorGreyScale
from wsipipe.preprocess.patching import GridPatchFinder, make_and_save_patchsets_for_dataset, load_patchsets_from_directory, combine
from wsipipe.preprocess.sample import balanced_sample
from pathlib import Path
import os

# Step 1: Define the path to your Camelyon16 data
cam16_path = Path("/workspace/data")

# Step 2: Create the training dataset object
train_dset = camelyon16.training(cam16_path)
print(train_dset.head())

# Step 3: Load the dataset using Camelyon16Loader
dset_loader = Camelyon16Loader()

# Step 4: Apply tissue detection
tisdet = TissueDetectorGreyScale(grey_level=0.85)

# Step 5: Create patchsets for the slides
patchfinder = GridPatchFinder(patch_level=0, patch_size=256, stride=256, labels_level=5)

# Define the output directory for the patchsets
path_to_pset_folder = Path("/workspace/data/sample_patchset")

# Create and save patchsets for the dataset
psets_for_dset = make_and_save_patchsets_for_dataset(
    dataset=train_dset,
    loader=dset_loader,
    tissue_detector=tisdet,
    patch_finder=patchfinder,
    output_dir=path_to_pset_folder
)

# Load the patchsets from the directory
psets_for_dset = load_patchsets_from_directory(patchsets_dir=path_to_pset_folder)

# Step 6: Combine patches from all slides into one dataset
all_patches_in_dset = combine(psets_for_dset)

# Step 7: Save all patches
path_to_all_patch_folder = Path("/workspace/data/all_patches")
all_patches_in_dset.export_patches(path_to_all_patch_folder)

# Step 8: Sample the patches
sampled_patches = balanced_sample(
    patches=all_patches_in_dset,
    num_samples=125000,  # 125,000 patches per class
    floor_samples=1000   # Minimum number of samples for each class
)

# Step 9: Save the sampled patches
path_to_sampled_patch_folder = Path("/workspace/data/sam_patch")
sampled_patches.export_patches(path_to_sampled_patch_folder)

print("Patch extraction and organization complete.")

# import os
# from pathlib import Path
# from wsipipe.datasets import camelyon16
# from wsipipe.load.datasets.camelyon16 import Camelyon16Loader
# from wsipipe.preprocess.tissue_detection import TissueDetectorGreyScale
# from wsipipe.preprocess.patching import GridPatchFinder, make_and_save_patchsets_for_dataset, combine
# from wsipipe.preprocess.sample import balanced_sample
# from tqdm import tqdm

# # Define paths and dataset
# cam16_path = Path("/workspace/data")
# train_dset = camelyon16.training(cam16_path)
# dset_loader = Camelyon16Loader()

# # Define simple tissue detector
# tisdet = TissueDetectorGreyScale(grey_level=0.85)

# # Define patch finder with additional options
# patchfinder = GridPatchFinder(
#     labels_level=5,
#     patch_level=0,
#     patch_size=256,
#     stride=256,
#     jitter=8,
#     remove_background=True
# )

# # Define output directory for patchsets and create it if it doesn't exist
# path_to_pset_folder = Path("/workspace/data/new_patchset")
# path_to_pset_folder.mkdir(parents=True, exist_ok=True)

# # Create and save patchsets for the dataset
# print("Creating and saving patchsets for the dataset...")
# psets_for_dset = make_and_save_patchsets_for_dataset(
#     dataset=train_dset,
#     loader=dset_loader,
#     tissue_detector=tisdet,
#     patch_finder=patchfinder,
#     output_dir=path_to_pset_folder
# )
# print("Patchsets creation and saving completed.")

# # Combine all patchsets into one
# print("Combining all patchsets into one...")
# all_patches_in_dset = combine(psets_for_dset)
# print("Combining patchsets completed.")

# # Export the full patches
# path_to_full_patches_folder = Path("/workspace/data/full_patches")
# path_to_full_patches_folder.mkdir(parents=True, exist_ok=True)
# print("Exporting full patches...")
# all_patches_in_dset.export_patches(path_to_full_patches_folder)
# print("Exporting full patches completed.")

# # Sample 250,000 patches in total, balanced across classes
# print("Sampling 250,000 patches, balanced across classes...")
# sampled_patches = balanced_sample(
#     patches=all_patches_in_dset,
#     num_samples=125000,  # 125,000 patches per class
#     floor_samples=1000   # Minimum number of samples for each class
# )
# print("Sampling completed.")

# # Define output directory for the sampled patches and create it if it doesn't exist
# path_to_sampled_patches_folder = Path("/workspace/data/sample_patches")
# path_to_sampled_patches_folder.mkdir(parents=True, exist_ok=True)

# # Export the sampled patches
# print("Exporting sampled patches...")
# sampled_patches.export_patches(path_to_sampled_patches_folder)
# print("Exporting sampled patches completed.")

# print("Full patches and sampled patches exported successfully.")

