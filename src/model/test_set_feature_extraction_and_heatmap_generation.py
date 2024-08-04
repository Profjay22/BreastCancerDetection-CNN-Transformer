import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Load the test set CSV
test_csv_path = "/data/ja235/camelyon16_project/test_tumor_probabilities/all_patches_inference_results.csv"
df = pd.read_csv(test_csv_path)

# Create output directories
output_dir = "/data/ja235/camelyon16_project/test_results"
heatmap_dir = os.path.join(output_dir, "tumor_heatmaps")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(heatmap_dir, exist_ok=True)

# Get the unique slide IDs
slide_ids = df['slide_id'].unique()

def process_tile(tile, tissue_mask_tile):
    tumor_mask_50 = tile > 0.5
    tumor_mask_90 = tile > 0.9
    
    labeled_regions_50 = label(tumor_mask_50)
    props_50 = regionprops(labeled_regions_50)
    
    tile_features = {
        'tumor_area_50': np.sum(tumor_mask_50),
        'tumor_area_90': np.sum(tumor_mask_90),
        'tissue_area': np.sum(tissue_mask_tile),
        'high_prob_pixel_count': np.sum(tumor_mask_90),
        'avg_prediction': np.mean(tile[tissue_mask_tile]) if np.sum(tissue_mask_tile) > 0 else 0
    }
    
    if props_50:
        largest_region = max(props_50, key=lambda r: r.area)
        tile_features.update({
            'longest_axis': max(largest_region.major_axis_length, largest_region.minor_axis_length),
            'largest_tumor_area': largest_region.area,
            'largest_tumor_eccentricity': largest_region.eccentricity,
            'largest_tumor_extent': largest_region.extent,
            'largest_tumor_perimeter': largest_region.perimeter,
            'largest_tumor_solidity': largest_region.solidity,
            'largest_tumor_convex_area': largest_region.convex_area
        })
    else:
        tile_features.update({
            'longest_axis': 0,
            'largest_tumor_area': 0,
            'largest_tumor_eccentricity': 0,
            'largest_tumor_extent': 0,
            'largest_tumor_perimeter': 0,
            'largest_tumor_solidity': 0,
            'largest_tumor_convex_area': 0
        })
    
    return tile_features

def extract_features(df_slide, slide_id):
    features = {}
    
    max_x = df_slide['x_coord'].max()
    max_y = df_slide['y_coord'].max()
    
    # Use the actual patch size from the data
    patch_size = 256  # Assuming 256x256 patches, adjust if different
    
    # Process the slide in smaller chunks
    chunk_size = 5000
    total_tumor_area_50 = 0
    total_tumor_area_90 = 0
    total_tissue_area = 0
    total_high_prob_pixels = 0
    total_prediction_sum = 0
    max_longest_axis = 0
    largest_tumor_area = 0
    largest_tumor_eccentricity = 0
    largest_tumor_extent = 0
    largest_tumor_perimeter = 0
    largest_tumor_solidity = 0
    largest_tumor_convex_area = 0
    
    all_tumor_properties = {
        'area': [],
        'perimeter': [],
        'eccentricity': [],
        'extent': [],
        'solidity': []
    }
    
    # Create low-resolution heatmap
    heatmap_resolution = 1000
    heatmap = np.zeros((heatmap_resolution, heatmap_resolution))
    
    for y in range(0, max_y + patch_size, chunk_size):
        for x in range(0, max_x + patch_size, chunk_size):
            chunk_df = df_slide[(df_slide['x_coord'] >= x) & (df_slide['x_coord'] < x + chunk_size) &
                                (df_slide['y_coord'] >= y) & (df_slide['y_coord'] < y + chunk_size)]
            
            if chunk_df.empty:
                continue
            
            # Create a chunk of the tumor probability map
            chunk_map = np.zeros((chunk_size, chunk_size))
            for _, row in chunk_df.iterrows():
                cx, cy = int(row['x_coord'] - x), int(row['y_coord'] - y)
                chunk_map[cy:cy+patch_size, cx:cx+patch_size] = row['tumor_probability']
            
            # Process the chunk
            tissue_mask_chunk = chunk_map > 0
            chunk_features = process_tile(chunk_map, tissue_mask_chunk)
            
            # Update feature aggregations
            total_tumor_area_50 += chunk_features['tumor_area_50']
            total_tumor_area_90 += chunk_features['tumor_area_90']
            total_tissue_area += chunk_features['tissue_area']
            total_high_prob_pixels += chunk_features['high_prob_pixel_count']
            total_prediction_sum += chunk_features['avg_prediction'] * chunk_features['tissue_area']
            
            if chunk_features['largest_tumor_area'] > largest_tumor_area:
                largest_tumor_area = chunk_features['largest_tumor_area']
                largest_tumor_eccentricity = chunk_features['largest_tumor_eccentricity']
                largest_tumor_extent = chunk_features['largest_tumor_extent']
                largest_tumor_perimeter = chunk_features['largest_tumor_perimeter']
                largest_tumor_solidity = chunk_features['largest_tumor_solidity']
                largest_tumor_convex_area = chunk_features['largest_tumor_convex_area']
                max_longest_axis = chunk_features['longest_axis']
            
            for prop in all_tumor_properties:
                all_tumor_properties[prop].append(chunk_features[f'largest_tumor_{prop}'])
            
            # Update low-resolution heatmap
            h_start_y = int(y * heatmap_resolution / (max_y + patch_size))
            h_end_y = int((y + chunk_size) * heatmap_resolution / (max_y + patch_size))
            h_start_x = int(x * heatmap_resolution / (max_x + patch_size))
            h_end_x = int((x + chunk_size) * heatmap_resolution / (max_x + patch_size))
            heatmap[h_start_y:h_end_y, h_start_x:h_end_x] = np.mean(chunk_map)
    
    # Calculate features
    features['tumor_tissue_ratio_50'] = total_tumor_area_50 / total_tissue_area if total_tissue_area > 0 else 0
    features['tumor_tissue_ratio_90'] = total_tumor_area_90 / total_tissue_area if total_tissue_area > 0 else 0
    features['longest_axis_largest_tumor'] = max_longest_axis
    features['high_prob_pixel_count'] = total_high_prob_pixels
    features['avg_prediction'] = total_prediction_sum / total_tissue_area if total_tissue_area > 0 else 0
    features['largest_tumor_area'] = largest_tumor_area
    features['largest_tumor_eccentricity'] = largest_tumor_eccentricity
    features['largest_tumor_extent'] = largest_tumor_extent
    features['largest_tumor_perimeter'] = largest_tumor_perimeter
    features['largest_tumor_solidity'] = largest_tumor_solidity
    features['tumor_convexity'] = largest_tumor_area / largest_tumor_convex_area if largest_tumor_convex_area > 0 else 0
    
    # Calculate additional statistical features
    for prop in all_tumor_properties:
        values = all_tumor_properties[prop]
        if values:
            features[f'{prop}_mean'] = np.mean(values)
            features[f'{prop}_max'] = np.max(values)
            features[f'{prop}_variance'] = np.var(values)
            features[f'{prop}_skewness'] = skew(values)
            features[f'{prop}_kurtosis'] = kurtosis(values)
        else:
            features[f'{prop}_mean'] = 0
            features[f'{prop}_max'] = 0
            features[f'{prop}_variance'] = 0
            features[f'{prop}_skewness'] = 0
            features[f'{prop}_kurtosis'] = 0
    
    # Save low-resolution heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap, cmap='jet', interpolation='nearest')
    plt.colorbar(label='Tumor Probability')
    plt.title(f'Tumor Probability Heatmap for Slide {slide_id}')
    plt.savefig(os.path.join(heatmap_dir, f'{slide_id}_heatmap.png'))
    plt.close()
    
    return features

slide_features = []

print(f"Processing {len(slide_ids)} slides...")
for slide_id in tqdm(slide_ids, desc="Processing slides"):
    try:
        slide_df = df[df['slide_id'] == slide_id]
        features = extract_features(slide_df, slide_id)
        features['slide_id'] = slide_id
        slide_features.append(features)
        
        # Save intermediate results every 10 slides
        if len(slide_features) % 10 == 0:
            intermediate_df = pd.DataFrame(slide_features)
            intermediate_df.to_csv(os.path.join(output_dir, 'intermediate_features.csv'), index=False)
            print(f"Saved intermediate results for {len(slide_features)} slides")
    except Exception as e:
        print(f"Error processing slide {slide_id}: {str(e)}")
        continue  # Skip to the next slide if there's an error

# Save final results
slide_features_df = pd.DataFrame(slide_features)
slide_features_df.to_csv(os.path.join(output_dir, 'test_slide_level_features.csv'), index=False)

print("Feature extraction completed. Results saved to:")
print(f"1. Heatmaps: {heatmap_dir}/<slide_id>_heatmap.png")
print(f"2. Features: {os.path.join(output_dir, 'test_slide_level_features.csv')}")