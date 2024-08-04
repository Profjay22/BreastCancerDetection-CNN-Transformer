import pandas as pd

# Define the paths to the CSV files
patch_positions_csv_path = "/data/ja235/camelyon16_project/fullpatch_positions.csv"
probabilities_csv_path = "/data/ja235/camelyon16_project/tumor_probabilities/tumor_probabilities.csv"
output_csv_path = "/data/ja235/camelyon16_project/merged_patch_probabilities.csv"

# Load the CSV files
patch_positions_df = pd.read_csv(patch_positions_csv_path)
probabilities_df = pd.read_csv(probabilities_csv_path)

# Ensure both dataframes have the same length
assert len(patch_positions_df) == len(probabilities_df), "Mismatch in number of patches"

# Add the tumor probability to the patch positions dataframe
patch_positions_df['tumor_probability'] = probabilities_df['tumor_probability']

# Save the combined dataframe to a new CSV file
patch_positions_df.to_csv(output_csv_path, index=False)

print(f"Merged data saved to {output_csv_path}")
