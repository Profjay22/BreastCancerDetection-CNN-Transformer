import pandas as pd
from sklearn.model_selection import train_test_split

# Load the slide-level features
slide_features_csv = "/data/ja235/camelyon16_project/tumor_heatmaps/slide_level_features.csv"
slide_features_df = pd.read_csv(slide_features_csv)

# Add a column for labels
slide_features_df['label'] = slide_features_df['slide_id'].apply(lambda x: 'tumor' if 'tumor' in x else 'normal')

# Define features and labels
X = slide_features_df.drop(columns=['slide_id', 'label'])
y = slide_features_df['label']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Combine features and labels for training set
train_df = X_train.copy()
train_df['label'] = y_train
train_df['slide_id'] = slide_features_df.loc[X_train.index, 'slide_id']

# Combine features and labels for validation set
val_df = X_val.copy()
val_df['label'] = y_val
val_df['slide_id'] = slide_features_df.loc[X_val.index, 'slide_id']

# Save the training set to CSV
train_csv_path = "/data/ja235/camelyon16_project/tumor_heatmaps/training_set.csv"
train_df.to_csv(train_csv_path, index=False)

# Save the validation set to CSV
val_csv_path = "/data/ja235/camelyon16_project/tumor_heatmaps/validation_set.csv"
val_df.to_csv(val_csv_path, index=False)

print(f"Training set saved to {train_csv_path}")
print(f"Validation set saved to {val_csv_path}")
