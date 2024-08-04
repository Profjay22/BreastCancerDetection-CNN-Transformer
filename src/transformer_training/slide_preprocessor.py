import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def preprocess_slides(input_dir, output_dir, max_patches_per_slide=3000, memory_load=0.6):
    os.makedirs(output_dir, exist_ok=True)

    all_npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])
    first_batch_size = int(len(all_npz_files) * memory_load)

    slide_data = defaultdict(lambda: {'label': None, 'features': [], 'slide_ids': []})

    def process_and_save_slide(slide_key, data):
        features = np.array(data['features'][:max_patches_per_slide])
        slide_ids = np.array(data['slide_ids'][:max_patches_per_slide])
        label = data['label']
        
        # Pad if necessary
        if len(features) < max_patches_per_slide:
            padding_size = max_patches_per_slide - len(features)
            features = np.pad(features, ((0, padding_size), (0, 0)), mode='constant')
            slide_ids = np.pad(slide_ids, (0, padding_size), mode='edge')
        
        # Create array for labels
        labels = np.full(max_patches_per_slide, label)
        
        # Save this slide's data
        np.savez_compressed(
            os.path.join(output_dir, f"slide_{slide_key}_label_{label}.npz"),
            features=features,
            slide_ids=slide_ids,
            labels=labels
        )

    def process_batch(batch_files):
        for file in tqdm(batch_files, desc="Processing NPZ files"):
            with np.load(os.path.join(input_dir, file)) as data:
                for slide_id, label, feature in zip(data['slide_ids'], data['labels'], data['features']):
                    slide_key = slide_id  # Use slide_id directly as the key
                    
                    if slide_data[slide_key]['label'] is None:
                        slide_data[slide_key]['label'] = 1 if 'tumor' in label else 0
                    
                    slide_data[slide_key]['features'].append(feature)
                    slide_data[slide_key]['slide_ids'].append(slide_id)
                    
                    # Check if we have enough features for this slide
                    if len(slide_data[slide_key]['features']) >= max_patches_per_slide:
                        process_and_save_slide(slide_key, slide_data[slide_key])
                        del slide_data[slide_key]

    print("Processing first 60% of NPZ files...")
    process_batch(all_npz_files[:first_batch_size])
    
    print("Processing remaining 40% of NPZ files...")
    process_batch(all_npz_files[first_batch_size:])

    print("Processing and saving remaining slides...")
    for slide_key, data in tqdm(list(slide_data.items()), desc="Saving remaining slides"):
        process_and_save_slide(slide_key, data)
        del slide_data[slide_key]

    print(f"Total slides processed: {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    input_dir = '/data/ja235/camelyon16_project/final_extracted_features'
    output_dir = '/data/ja235/camelyon16_project/check_preprocessed_slides'
    preprocess_slides(input_dir, output_dir)