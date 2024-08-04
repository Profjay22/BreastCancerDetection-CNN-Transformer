import os
import time
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import psutil

# CUDA setup
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Define data paths
workspace_dir = Path("/data/ja235/camelyon16_project/")
train_patches_dir = workspace_dir / "data/augmented_patches/train"
val_patches_dir = workspace_dir / "data/augmented_patches/val"
processed_train_dir = workspace_dir / "data/processed/train"
processed_val_dir = workspace_dir / "data/processed/val"

# Create directories for processed data if they don't exist
processed_train_dir.mkdir(parents=True, exist_ok=True)
processed_val_dir.mkdir(parents=True, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.9, saturation=0.25, hue=0.04),
    transforms.ToTensor(),
])

# GPU Normalization
class GPUNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, tensor):
        return (tensor - self.mean) / self.std

gpu_normalize = GPUNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def preprocess_dataset(input_dir, output_dir, initial_batch_size=64):
    print(f"Step 1: Initializing dataset from {input_dir}")
    dataset = ImageFolder(input_dir, transform=transform)
    
    batch_size = initial_batch_size
    print(f"Step 2: Creating DataLoader with initial batch size {batch_size}")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
    
    start_time = time.time()
    total_images = len(dataset)
    total_batches = len(dataloader)
    
    print(f"Step 3: Starting preprocessing for {total_images} images ({total_batches} batches)")
    
    # Create a GradScaler for mixed precision
    scaler = amp.GradScaler()
    
    for i, (images, labels) in enumerate(dataloader):
        batch_start_time = time.time()
        
        print(f"  Processing batch {i+1}/{total_batches}")
        
        try:
            # Move data to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Use mixed precision
            with amp.autocast():
                # Apply normalization
                images = gpu_normalize(images)
            
            # Move data back to CPU for saving
            images = images.cpu()
            labels = labels.cpu()
            
            torch.save((images, labels), output_dir / f"batch_{i:05d}.pt")
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            images_processed = (i + 1) * batch_size
            elapsed_time = batch_end_time - start_time
            
            print(f"    Batch {i+1} completed in {batch_duration:.2f} seconds")
            print(f"    Progress: {images_processed}/{total_images} images "
                  f"({images_processed/total_images*100:.2f}%)")
            print(f"    Total time elapsed: {elapsed_time:.2f} seconds")
            print(f"    Current memory usage: {psutil.virtual_memory().percent}%")
            print(f"    Current GPU memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            # Force CUDA to synchronize to get accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # If processing is going smoothly, try increasing batch size
            if i > 0 and i % 10 == 0 and torch.cuda.memory_allocated() < 0.7 * torch.cuda.get_device_properties(0).total_memory:
                batch_size = min(batch_size * 2, 256)  # Cap at 256
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
                print(f"    Increased batch size to {batch_size}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    WARNING: Out of GPU memory. Reducing batch size and retrying.")
                torch.cuda.empty_cache()
                batch_size = max(batch_size // 2, 1)  # Ensure batch size doesn't go below 1
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
                continue
            else:
                raise e
        
        print()  # Empty line for readability

    print(f"Step 4: Finished processing {total_images} images in {time.time() - start_time:.2f} seconds")

def main():
    torch.backends.cudnn.benchmark = True  # This can speed up training

    print("Level 1: Preprocessing training data...")
    preprocess_dataset(train_patches_dir, processed_train_dir)

    print("\nLevel 2: Preprocessing validation data...")
    preprocess_dataset(val_patches_dir, processed_val_dir)

    print("\nLevel 3: Preprocessing complete.")

if __name__ == "__main__":
    main()