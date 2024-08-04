import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('/data/ja235/camelyon16_project/tumor_heatmaps/slide_level_features.csv')

# Extract labels from slide_id
data['label'] = data['slide_id'].apply(lambda x: 'tumor' if 'tumor' in x else 'normal')

# Summary statistics
summary_stats = data.groupby('label').describe()

# Save summary statistics to a CSV file
summary_stats.to_csv('summary_statistics_by_label.csv')
print("Summary statistics saved to summary_statistics_by_label.csv")

# Visualize distributions for a few selected features
features_to_visualize = ['tumor_tissue_ratio_50', 'tumor_tissue_ratio_90', 'avg_prediction']

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features_to_visualize):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data, x=feature, hue='label', kde=True, element='step')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
print("Feature distributions saved to feature_distributions.png")
