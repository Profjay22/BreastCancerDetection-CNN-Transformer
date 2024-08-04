import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

# Function to login to Hugging Face
def login_to_huggingface(token):
    try:
        login(token, add_to_git_credential=True)
        print("Successfully logged in to Hugging Face.")
        return True
    except Exception as e:
        print(f"Failed to log in: {e}")
        return False

# Function to download and load the model
def download_and_load_model(local_dir, model_name):
    try:
        os.makedirs(local_dir, exist_ok=True)  # Create directory if it does not exist
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model = timm.create_model(
            model_name, img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

# Main code execution
if login_to_huggingface("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"):
    local_dir = "/home/ja235/.cache/huggingface/models--MahmoodLab--UNI/"
    model_name = "vit_large_patch16_224"
    model = download_and_load_model(local_dir, model_name)
else:
    print("Login failed. Please check your token.")
