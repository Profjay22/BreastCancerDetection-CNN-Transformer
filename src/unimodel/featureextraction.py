import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import psutil
import time
import gc
import json
from datetime import datetime

# Step 1: Login to HuggingFace
def hf_login(access_token):
    try:
        login(token=access_token, add_to_git_credential=True)
        print("Successfully logged in to HuggingFace")
    except Exception as e:
        print(f"Failed to login to HuggingFace: {e}")
        raise

# Step 2: Load the UNI Model and Transforms
def load_uni_model():
    try:
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**config)
        model.eval()
        model.to('cuda')  # Move model to GPU
        print("UNI model loaded successfully")
        return model, transform
    except Exception as e:
        print(f"Failed to load UNI model: {e}")
        raise

# Step 3: Create a Dataset Class for Patches
class PatchDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        slide_id, x_coord, y_coord, label = self.parse_filename(image_path)
        return image, slide_id, x_coord, y_coord, label

    @staticmethod
    def parse_filename(filename):
        basename = os.path.basename(filename)
        parts = basename.split('-')
        slide_id = parts[0].split('_')[1]
        x_coord = int(parts[1])
        y_coord = int(parts[2])
        label = parts[0].split('_')[0]  # Extract the label from the prefix 'normal' or 'tumor'
        return slide_id, x_coord, y_coord, label

# Step 4: Load Patches from Directories
def load_patches_from_directory(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Step 5: Extract Features and Save
def extract_features_and_save(model, data_loader, output_dir, batch_size, save_interval=10000, resume_from=0, start_time=None):
    os.makedirs(output_dir, exist_ok=True)
    all_features = []
    slide_ids = []
    x_coords = []
    y_coords = []
    labels = []

    total_patches = 0
    total_batches = len(data_loader)
    progress_bar = tqdm(total=total_batches, initial=resume_from//batch_size, desc="Extracting features")

    if start_time is None:
        start_time = datetime.now()
    else:
        start_time = datetime.fromisoformat(start_time)

    try:
        with torch.no_grad():
            for batch_idx, (images, batch_slide_ids, batch_x_coords, batch_y_coords, batch_labels) in enumerate(data_loader):
                if batch_idx * batch_size < resume_from:
                    continue  # Skip already processed batches

                images = images.cuda()  # Move images to GPU
                batch_features = model(images).cpu().numpy()
                all_features.append(batch_features)
                slide_ids.extend(batch_slide_ids)
                x_coords.extend(batch_x_coords)
                y_coords.extend(batch_y_coords)
                labels.extend(batch_labels)

                total_patches += len(images)
                progress_bar.update(1)

                # Check CPU usage and sleep if necessary
                if psutil.cpu_percent() > 90:
                    time.sleep(5)

                # Clear GPU cache if memory is getting full
                if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
                    torch.cuda.empty_cache()

                # Save features at regular intervals
                if total_patches >= save_interval:
                    save_features(all_features, slide_ids, x_coords, y_coords, labels, output_dir, batch_idx)
                    all_features, slide_ids, x_coords, y_coords, labels = [], [], [], [], []
                    total_patches = 0

                    # Save progress with elapsed time
                    elapsed_time = datetime.now() - start_time
                    save_progress(output_dir, (batch_idx + 1) * batch_size, start_time.isoformat(), str(elapsed_time))

        progress_bar.close()

        # Save any remaining features
        if all_features:
            save_features(all_features, slide_ids, x_coords, y_coords, labels, output_dir, 'final')
            # Save final progress
            elapsed_time = datetime.now() - start_time
            save_progress(output_dir, total_patches, start_time.isoformat(), str(elapsed_time))

        print(f"Feature extraction complete. Total time elapsed: {elapsed_time}. All features saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        raise

def save_features(features, slide_ids, x_coords, y_coords, labels, output_dir, batch_idx):
    features = np.concatenate(features, axis=0)
    slide_ids = np.array(slide_ids)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    labels = np.array(labels)

    output_path = os.path.join(output_dir, f"features_batch_{batch_idx}.npz")
    np.savez_compressed(output_path, 
                        features=features, 
                        slide_ids=slide_ids, 
                        x_coords=x_coords, 
                        y_coords=y_coords,
                        labels=labels)
    print(f"Saved features batch {batch_idx} to {output_path}")

def save_progress(output_dir, processed_patches, start_time, elapsed_time):
    progress_file = os.path.join(output_dir, "progress.json")
    progress_data = {
        "processed_patches": processed_patches,
        "start_time": start_time,
        "elapsed_time": elapsed_time
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def load_progress(output_dir):
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        processed_patches = progress.get("processed_patches", 0)
        start_time = progress.get("start_time", None)
        return processed_patches, start_time
    return 0, None

def get_optimal_batch_size(initial_batch_size=32, max_batch_size=64):
    while initial_batch_size <= max_batch_size:
        try:
            # Try to create a batch of random data
            torch.rand((initial_batch_size, 3, 224, 224)).cuda()
            return initial_batch_size
        except RuntimeError:
            # If we run out of memory, reduce batch size and try again
            initial_batch_size //= 2
            if initial_batch_size < 1:
                raise ValueError("Unable to find a suitable batch size")

if __name__ == "__main__":
    # Define paths
    normal_dir = '/data/ja235/camelyon16_project/data/full_all_patches/normal'
    tumor_dir = '/data/ja235/camelyon16_project/data/full_all_patches/tumor'
    output_dir = '/data/ja235/camelyon16_project/final_extracted_features'
    
    # HuggingFace Access Token
    access_token = "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"

    # Step 1: Login to HuggingFace
    hf_login(access_token)

    # Step 2: Load the UNI Model and Transforms
    model, transform = load_uni_model()

    # Determine optimal batch size
    optimal_batch_size = get_optimal_batch_size()
    print(f"Using optimal batch size: {optimal_batch_size}")

    # Load patches
    print("Loading patch paths...")
    normal_image_paths = load_patches_from_directory(normal_dir)
    tumor_image_paths = load_patches_from_directory(tumor_dir)
    all_image_paths = normal_image_paths + tumor_image_paths
    print(f"Total patches: {len(all_image_paths)}")

    # Create dataset and dataloader
    dataset = PatchDataset(all_image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load progress
    resume_from, start_time = load_progress(output_dir)
    print(f"Resuming from patch {resume_from}")

    # Step 5: Extract Features and Save
    extract_features_and_save(model, dataloader, output_dir, optimal_batch_size, save_interval=10000, resume_from=resume_from, start_time=start_time)

    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

    print("Process completed successfully.")
