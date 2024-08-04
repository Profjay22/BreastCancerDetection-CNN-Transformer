import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, ColorJitter
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from pathlib import Path

# Define data paths
workspace_dir = Path("/data/ja235/camelyon16_project/")
sampled_patches_dir = workspace_dir / "data/sam_patch"

# Define transformations
train_transform = Compose([
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    RandomCrop(224),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5))
])

# Load sampled dataset
sampled_dataset = ImageFolder(sampled_patches_dir, transform=train_transform)

# Split the dataset
train_size = int(0.8 * len(sampled_dataset))
val_size = len(sampled_dataset) - train_size
train_dataset, val_dataset = random_split(sampled_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

# Define the search space for Bayesian Optimization
search_space = [
    Real(1e-5, 1e-1, name='learning_rate'),
    Real(0.1, 0.5, name='dropout_rate'),
    Integer(16, 128, name='batch_size')
]

@use_named_args(search_space)
def objective(learning_rate, dropout_rate, batch_size):
    model = GoogleNetModel(dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    trainer = Trainer(
        max_epochs=3,  
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        logger=False,
        enable_checkpointing=False,
    )

    result = trainer.fit(model, train_loader, val_loader)

    # Use validation loss as the metric to minimize
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss

def run_bayesian_optimization():
    res = gp_minimize(objective, search_space, n_calls=20, random_state=0)
    print("Best hyperparameters:", res.x)
    plot_convergence(res)
    return res.x

if __name__ == "__main__":
    run_bayesian_optimization()
