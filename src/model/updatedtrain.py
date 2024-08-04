import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, ColorJitter
from torch.utils.data import DataLoader, random_split
from torchvision.models import googlenet, GoogLeNet_Weights
from pathlib import Path
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import matplotlib.pyplot as plt
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

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
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load sampled dataset
sampled_dataset = ImageFolder(sampled_patches_dir, transform=train_transform)

# Split the dataset
train_size = int(0.8 * len(sampled_dataset))
val_size = len(sampled_dataset) - train_size
train_dataset, val_dataset = random_split(sampled_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

class LRLossRecorder(Callback):
    def __init__(self):
        self.lrs = []
        self.losses = []
        self.iterations = []
        self.iteration_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.iteration_count += 1
        self.lrs.append(trainer.optimizers[0].param_groups[0]['lr'])
        self.losses.append(outputs['loss'].item())
        self.iterations.append(self.iteration_count)

class MetricsHistory(Callback):
    def __init__(self):
        self.metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics['train_loss'].append(trainer.callback_metrics['train_loss'].item())
        self.metrics['train_acc'].append(trainer.callback_metrics['train_acc'].item())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics['val_loss'].append(trainer.callback_metrics['val_loss'].item())
        self.metrics['val_acc'].append(trainer.callback_metrics['val_acc'].item())

class GoogleNetModel(LightningModule):
    def __init__(self):
        super(GoogleNetModel, self).__init__()
        weights = GoogLeNet_Weights.DEFAULT
        self.model = googlenet(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        acc = torch.sum(outputs.argmax(dim=1) == labels).float() / len(labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        acc = torch.sum(outputs.argmax(dim=1) == labels).float() / len(labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        scheduler = StepLR(optimizer, step_size=50000, gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

def plot_lr_loss(recorder, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(recorder.iterations, recorder.losses)
    plt.title('Learning Rate vs Training Loss')
    plt.xlabel('Training iterations')
    plt.ylabel('Training loss')
    
    lr_changes = [0] + [i for i in range(1, len(recorder.lrs)) if recorder.lrs[i] != recorder.lrs[i-1]]
    for i in lr_changes:
        plt.axvline(x=recorder.iterations[i], color='gray', linestyle='--')
        plt.text(recorder.iterations[i], plt.ylim()[1], f'{recorder.lrs[i]:.0e}', 
                 horizontalalignment='center', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"LR vs Loss plot saved to {save_path}")

def plot_metrics(metrics_history, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['train_acc'], label='Train Accuracy')
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy During Training')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics plot saved to {save_path}")

def log_system_memory():
    print(f"Current system memory usage: {psutil.virtual_memory().percent}%")

def monitor_resources():
    cpu_usage = psutil.cpu_percent(interval=1)
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_mem_info = nvmlDeviceGetMemoryInfo(handle)
    gpu_usage = gpu_mem_info.used / gpu_mem_info.total * 100
    return cpu_usage, gpu_usage

def adjust_parameters(cpu_usage, gpu_usage, batch_size, num_workers):
    if cpu_usage > 80 or gpu_usage > 80:
        time.sleep(5)
    else:
        if batch_size < 128:
            batch_size = int(batch_size * 1.5)
        if num_workers < 8:
            num_workers += 1
    return batch_size, num_workers

def main():
    start_time = time.time()
    batch_size = 32
    num_workers = 4

    cpu_usage, gpu_usage = monitor_resources()
    batch_size, num_workers = adjust_parameters(cpu_usage, gpu_usage, batch_size, num_workers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    logger = TensorBoardLogger(save_dir=os.path.join(workspace_dir, 'logs'), name='GoogleNetModel')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(workspace_dir, 'model'), 
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}', 
        save_top_k=1, 
        monitor='val_loss', 
        mode='min', 
        save_last=True  # Save the last model as well
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
    lr_loss_recorder = LRLossRecorder()
    metrics_history = MetricsHistory()

    trainer = Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_callback, early_stopping, lr_loss_recorder, metrics_history],
        logger=logger
    )

    model = GoogleNetModel()
    
    train_start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_end_time = time.time()

    total_training_time = train_end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    log_system_memory()

    plot_lr_loss(lr_loss_recorder, os.path.join(workspace_dir, 'lr_vs_loss.png'))
    plot_metrics(metrics_history.metrics, os.path.join(workspace_dir, 'training_metrics.png'))

if __name__ == "__main__":
    main()
