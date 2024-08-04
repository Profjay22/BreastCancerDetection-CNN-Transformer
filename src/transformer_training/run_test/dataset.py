import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SlideDataset(Dataset):
    def __init__(self, normal_dir, tumor_dir, max_patches_per_slide=1000, max_sequence_length=15000):
        self.normal_dir = normal_dir
        self.tumor_dir = tumor_dir
        self.max_patches_per_slide = max_patches_per_slide
        self.max_sequence_length = max_sequence_length
        self.slide_info = self.get_slide_info()
        self.slide_keys = list(self.slide_info.keys())

    def get_slide_info(self):
        slide_info = {}
        
        # Indexing normal files
        for file in tqdm(os.listdir(self.normal_dir), desc="Indexing normal NPZ files"):
            if file.endswith('.npz'):
                data = np.load(os.path.join(self.normal_dir, file))
                for slide_id in data['slide_ids']:
                    slide_id = str(slide_id)
                    if slide_id not in slide_info:
                        slide_info[slide_id] = {'label': 0, 'normal_files': [], 'tumor_files': []}
                    slide_info[slide_id]['normal_files'].append(os.path.join(self.normal_dir, file))

        # Indexing tumor files
        for file in tqdm(os.listdir(self.tumor_dir), desc="Indexing tumor NPZ files"):
            if file.endswith('.npz'):
                data = np.load(os.path.join(self.tumor_dir, file))
                for slide_id in data['slide_ids']:
                    slide_id = str(slide_id)
                    if slide_id not in slide_info:
                        slide_info[slide_id] = {'label': 1, 'normal_files': [], 'tumor_files': []}
                    slide_info[slide_id]['tumor_files'].append(os.path.join(self.tumor_dir, file))
        
        return slide_info

    def __len__(self):
        return len(self.slide_keys)

    def __getitem__(self, idx):
        slide_id = self.slide_keys[idx]
        slide_info = self.slide_info[slide_id]
        
        features = []
        tumor_positive_features = []
        tumor_negative_features = []
        
        # Load normal files
        for file in slide_info['normal_files']:
            data = np.load(file)
            mask = data['slide_ids'] == slide_id
            features.extend(data['features'][mask])

        # Load tumor files
        for file in slide_info['tumor_files']:
            data = np.load(file)
            mask = data['slide_ids'] == slide_id
            tumor_positive_features.extend(data['features'][mask])

        # Handle tumor slides
        if slide_info['label'] == 1:  # Tumor slide
            if len(tumor_positive_features) > int(self.max_patches_per_slide * 0.8):
                tumor_positive_features = tumor_positive_features[:int(self.max_patches_per_slide * 0.8)]
            tumor_negative_features = features[:int(self.max_patches_per_slide * 0.2)]
            features = tumor_positive_features + tumor_negative_features
        else:  # Normal slide
            if len(features) > self.max_patches_per_slide:
                features = features[:self.max_patches_per_slide]
        
        # Limit the sequence length
        if len(features) > self.max_sequence_length:
            features = features[:self.max_sequence_length]
        
        features = np.array(features)
        features = torch.tensor(features, dtype=torch.float32)
        slide_label = torch.tensor(slide_info['label'], dtype=torch.long)
        
        return features, slide_label, slide_id

def collate_fn(batch):
    features, labels, slide_keys = zip(*batch)
    max_len = max(feat.size(0) for feat in features)
    padded_features = torch.zeros(len(features), max_len, features[0].size(1))
    for i, feat in enumerate(features):
        padded_features[i, :feat.size(0), :] = feat
    labels = torch.stack(labels)
    return padded_features, labels, slide_keys
