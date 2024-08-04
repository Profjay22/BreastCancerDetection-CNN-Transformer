import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SlideDataset(Dataset):
    def __init__(self, feature_dir, max_patches_per_slide=1000, max_sequence_length=15000, limit_normal=True):
        self.feature_dir = feature_dir
        self.max_patches_per_slide = max_patches_per_slide
        self.max_sequence_length = max_sequence_length
        self.limit_normal = limit_normal
        self.tumor_positive_files = set([
            "features_batch_313938.npz",
            "features_batch_314251.npz",
            "features_batch_314564.npz",
            "features_batch_314877.npz",
            "features_batch_315190.npz",
            "features_batch_315503.npz",
            "features_batch_315816.npz",
            "features_batch_316129.npz",
            "features_batch_316442.npz",
            "features_batch_316755.npz",
            "features_batch_317068.npz",
            "features_batch_317381.npz",
            "features_batch_317382.npz"
        ])
        self.slide_info = self.get_slide_info()
        self.slide_keys = list(self.slide_info.keys())

    def get_slide_info(self):
        slide_info = {}
        normal_count = 0
        tumor_count = 0
        
        for file in tqdm(os.listdir(self.feature_dir), desc="Indexing NPZ files"):
            if file.endswith('.npz'):
                with np.load(os.path.join(self.feature_dir, file)) as data:
                    for slide_id, label in zip(data['slide_ids'], data['labels']):
                        slide_key = f"{slide_id}_{label.split('_')[0]}"
                        if slide_key not in slide_info:
                            slide_info[slide_key] = {
                                'label': 1 if 'tumor' in label else 0,
                                'files': set(),
                                'is_tumor_positive': file in self.tumor_positive_files
                            }
                        slide_info[slide_key]['files'].add(file)
        
        for slide_key, info in slide_info.items():
            if info['label'] == 0:
                normal_count += 1
            else:
                tumor_count += 1
        
        print(f"Total slides: {len(slide_info)}")
        print(f"Normal slides: {normal_count}")
        print(f"Tumor slides: {tumor_count}")
        
        return slide_info

    def __len__(self):
        return len(self.slide_keys)

    def __getitem__(self, idx):
        slide_key = self.slide_keys[idx]
        slide_info = self.slide_info[slide_key]
        
        features = []
        tumor_positive_features = []
        tumor_negative_features = []
        
        for file in slide_info['files']:
            data = np.load(os.path.join(self.feature_dir, file))
            mask = (data['slide_ids'] == slide_key.split('_')[0]) & np.char.startswith(data['labels'].astype(str), slide_key.split('_')[1])
            slide_features = data['features'][mask]
            
            if slide_info['label'] == 1:  # Tumor slide
                if file in self.tumor_positive_files:
                    tumor_positive_features.extend(slide_features)
                else:
                    tumor_negative_features.extend(slide_features)
            else:
                features.extend(slide_features)
        
        if slide_info['label'] == 1:  # Tumor slide
            if len(tumor_positive_features) > int(self.max_patches_per_slide * 0.8):
                tumor_positive_features = tumor_positive_features[:int(self.max_patches_per_slide * 0.8)]
            if len(tumor_negative_features) > int(self.max_patches_per_slide * 0.2):
                tumor_negative_features = tumor_negative_features[:int(self.max_patches_per_slide * 0.2)]
            features = tumor_positive_features + tumor_negative_features
        else:
            if len(features) > self.max_patches_per_slide:
                features = features[:self.max_patches_per_slide]
        
        # Limit the sequence length
        if len(features) > self.max_sequence_length:
            print(f"Limiting features for slide {slide_key} from {len(features)} to {self.max_sequence_length}")
            features = features[:self.max_sequence_length]
        
        features = np.array(features)  # Convert list of numpy arrays to a single numpy array
        features = torch.tensor(features, dtype=torch.float32)
        slide_label = torch.tensor(slide_info['label'], dtype=torch.long)
        
        return features, slide_label, slide_key

def create_stratified_split(dataset, test_size=0.2, random_state=42):
    slide_keys = list(dataset.slide_info.keys())
    labels = [info['label'] for info in dataset.slide_info.values()]
    
    train_indices, val_indices = train_test_split(
        range(len(slide_keys)),
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Debugging: Print sample slide IDs and labels
    print("Sample training slide IDs and labels:")
    for idx in train_indices[:5]:
        print(slide_keys[idx], labels[idx])
    
    print("Sample validation slide IDs and labels:")
    for idx in val_indices[:5]:
        print(slide_keys[idx], labels[idx])
    
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def collate_fn(batch):
    features, labels, slide_keys = zip(*batch)
    
    # Get the maximum sequence length in this batch
    max_len = max(feat.size(0) for feat in features)
    
    # Pad each feature tensor to the maximum length
    padded_features = torch.zeros(len(features), max_len, features[0].size(1))
    for i, feat in enumerate(features):
        padded_features[i, :feat.size(0), :] = feat
    
    labels = torch.stack(labels)
    return padded_features, labels, slide_keys