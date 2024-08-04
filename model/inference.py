import os
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from updatedtrain import GoogleNetModel
from PIL import Image
from tqdm import tqdm

# Define data paths
patches_dir = Path("/data/ja235/camelyon16_project/data/full_all_patches")
model_path = Path("/data/ja235/camelyon16_project/model/last.ckpt")

# Define transformations
transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.image_paths.extend(list((root_dir / 'normal__1').glob("*.png")))
        self.image_paths.extend(list((root_dir / 'tumor').glob("*.png")))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path)

# Function to extract slide ID, x, y from filename
def extract_info(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('-')
    slide_id = parts[0].split('_')[-1]
    x_coord = parts[1]
    y_coord = parts[2].split('.')[0]  # Remove file extension
    return slide_id, x_coord, y_coord

# Load dataset
dataset = CustomDataset(patches_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Load model
model = GoogleNetModel.load_from_checkpoint(model_path)
model.eval()

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prepare lists to store information
all_paths = []
all_slide_ids = []
all_x_coords = []
all_y_coords = []
all_probs = []

# Run inference and extract information
print("Starting inference...")
with torch.no_grad():
    for images, paths in tqdm(data_loader, desc="Processing patches"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of tumor class
        
        all_paths.extend(paths)
        all_probs.extend(probs.cpu().numpy())

        for img_path in paths:
            slide_id, x_coord, y_coord = extract_info(img_path)
            all_slide_ids.append(slide_id)
            all_x_coords.append(x_coord)
            all_y_coords.append(y_coord)

# Convert to DataFrame
df_info = pd.DataFrame({
    "patch_path": all_paths,
    "slide_id": all_slide_ids,
    "x_coord": all_x_coords,
    "y_coord": all_y_coords,
    "tumor_probability": all_probs
})

print("Inference completed. Total patches processed:", len(df_info))

# Save to CSV
output_dir = Path("/data/ja235/camelyon16_project/train_tumor_probabilities")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "train_patches_inference.csv"
df_info.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Display a sample of the results
print("\nSample of the results:")
print(df_info.head())
print("\nDataFrame shape:", df_info.shape)