import torch
from pathlib import Path

# Define the directory where the processed data is saved
processed_train_dir = Path("/data/ja235/camelyon16_project/data/processed/train")

# Verify each saved batch
def verify_saved_batches(processed_dir):
    batch_files = list(processed_dir.glob("batch_*.pt"))
    all_correct = True
    for batch_file in batch_files:
        try:
            images, labels, coordinates = torch.load(batch_file)
            
            # Check if images, labels, and coordinates lengths match
            if len(images) != len(labels) or len(images) != len(coordinates):
                print(f"Mismatch in {batch_file}: images ({len(images)}), labels ({len(labels)}), coordinates ({len(coordinates)})")
                all_correct = False
            else:
                print(f"{batch_file} verified successfully.")
        
        except Exception as e:
            print(f"Error loading {batch_file}: {e}")
            all_correct = False
    
    if all_correct:
        print("All batches verified successfully.")
    else:
        print("Some batches had issues. Please review the log for details.")

if __name__ == "__main__":
    verify_saved_batches(processed_train_dir)
