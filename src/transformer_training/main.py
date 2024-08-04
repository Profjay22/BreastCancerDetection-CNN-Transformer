from dataset import SlideDataset, create_stratified_split, collate_fn
from model import TransformerForSlideClassification
from train import Trainer
import torch
import pandas as pd
from torch.utils.data import DataLoader

if _name_ == "_main_":
    feature_dir = '/data/ja235/camelyon16_project/final_extracted_features'
    dataset = SlideDataset(feature_dir, max_patches_per_slide=1000, limit_normal=True)
    
    train_dataset, val_dataset = create_stratified_split(dataset)
    
    # Debugging: Check label distribution
    train_labels = [dataset.slide_info[dataset.slide_keys[i]]['label'] for i in train_dataset.indices]
    val_labels = [dataset.slide_info[dataset.slide_keys[i]]['label'] for i in val_dataset.indices]

    print("Training label distribution:", pd.Series(train_labels).value_counts())
    print("Validation label distribution:", pd.Series(val_labels).value_counts())
    
    # Debugging: Check if there is overlap between train and validation sets
    train_slide_keys = set(dataset.slide_keys[i] for i in train_dataset.indices)
    val_slide_keys = set(dataset.slide_keys[i] for i in val_dataset.indices)

    # Ensure there is no overlap
    assert train_slide_keys.isdisjoint(val_slide_keys), "Data leakage detected between train and validation sets!"
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    model = TransformerForSlideClassification()
    trainer = Trainer(model, train_loader, val_loader, test_loader=None)  # Assuming no test loader for now
    trainer.train()