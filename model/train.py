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
    Normalize((0.5, 0.5, 0.5))
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
    def __init__(self, dropout_rate=0.5):
        super(GoogleNetModel, self).__init__()
        weights = GoogLeNet_Weights.DEFAULT
        self.model = googlenet(weights=weights)
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, 2)
        )
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
    from bayesian_optimization import run_bayesian_optimization

    # Run Bayesian optimization to find the best hyperparameters
    best_hyperparameters = run_bayesian_optimization()
    learning_rate, dropout_rate, batch_size = best_hyperparameters

    start_time = time.time()
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

    model = GoogleNetModel(dropout_rate=dropout_rate)  # Pass dropout rate to the model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Use optimized learning rate

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



# import os
# import time
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.optim.lr_scheduler import ExponentialLR
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, precision_score, recall_score, f1_score
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, ColorJitter
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.models import googlenet, GoogLeNet_Weights
# from pathlib import Path
# from tqdm import tqdm
# from PIL import ImageEnhance, Image

# # Custom transformation to enhance visibility
# class EnhanceVisibility(object):
#     def __init__(self, factor):
#         self.factor = factor

#     def __call__(self, img):
#         enhancer = ImageEnhance.Contrast(img)
#         img = enhancer.enhance(self.factor)
#         return img

# # Define data paths
# workspace_dir = Path("/workspace")
# train_patches_dir = workspace_dir / "data/all_patches/train"
# val_patches_dir = workspace_dir / "data/all_patches/val"
# sampled_patches_dir = workspace_dir / "data/sampled_patches"

# # Define transformations
# train_transform = Compose([
#     RandomCrop((224, 224)),
#     EnhanceVisibility(factor=1.5),
#     ColorJitter(brightness=0.25, contrast=0.9, saturation=0.25, hue=0.04),
#     ToTensor(),
#     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# val_transform = Compose([
#     ToTensor(),
#     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Load datasets
# print("Loading datasets...")
# train_dataset = ImageFolder(train_patches_dir, transform=train_transform)
# val_dataset = ImageFolder(val_patches_dir, transform=val_transform)
# sampled_dataset = ImageFolder(sampled_patches_dir, transform=val_transform)

# # Create data loaders
# print("Creating data loaders...")
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# # Initialize the model
# print("Initializing model...")
# weights = GoogLeNet_Weights.DEFAULT
# model = googlenet(weights=weights)
# model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming 2 classes: normal and tumor
# model = model.cuda()  # Use GPU

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = ExponentialLR(optimizer, gamma=0.97)  # Exponential decay for the learning rate

# # Early stopping parameters
# patience = 10
# best_loss = float('inf')
# best_model_wts = None
# counter = 0

# # Logging with TensorBoard
# log_dir = workspace_dir / "logs"
# os.makedirs(log_dir, exist_ok=True)
# writer = SummaryWriter(log_dir)

# # Lists to store loss and accuracy
# train_losses = []
# val_losses = []
# train_accuracies = []
# val_accuracies = []

# # Training function
# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100):
#     global best_loss, best_model_wts, counter
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#                 data_loader = train_loader
#             else:
#                 model.eval()   # Set model to evaluate mode
#                 data_loader = val_loader

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data
#             for inputs, labels in tqdm(data_loader, desc=f"{phase} Epoch {epoch+1}"):
#                 inputs = inputs.cuda()
#                 labels = labels.cuda()

#                 # Zero the parameter gradients
#                 optimizer.zero_grad()

#                 # Forward
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # Backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         # Gradient clipping
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                         optimizer.step()

#                 # Statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(data_loader.dataset)
#             epoch_acc = running_corrects.double() / len(data_loader.dataset)

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             if phase == 'train':
#                 train_losses.append(epoch_loss)
#                 train_accuracies.append(epoch_acc)
#                 writer.add_scalar('Loss/train', epoch_loss, epoch)
#                 writer.add_scalar('Accuracy/train', epoch_acc, epoch)
#             else:
#                 val_losses.append(epoch_loss)
#                 val_accuracies.append(epoch_acc)
#                 writer.add_scalar('Loss/val', epoch_loss, epoch)
#                 writer.add_scalar('Accuracy/val', epoch_acc, epoch)

#                 # Deep copy the model
#                 if epoch_loss < best_loss:
#                     best_loss = epoch_loss
#                     best_model_wts = model.state_dict()
#                     counter = 0
#                 else:
#                     counter += 1

#                 if counter >= patience:
#                     print("Early stopping")
#                     model.load_state_dict(best_model_wts)
#                     return model

#         scheduler.step()

#     model.load_state_dict(best_model_wts)
#     return model

# # Function to identify hard negatives and save probabilities
# def find_hard_negatives(model, data_loader):
#     hard_negatives = []
#     probabilities = []
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in tqdm(data_loader, desc="Finding Hard Negatives"):
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             outputs = model(inputs)
#             probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of tumor class
#             _, preds = torch.max(outputs, 1)
#             for i in range(len(labels)):
#                 if preds[i] != labels[i]:
#                     hard_negatives.append((inputs[i].cpu(), labels[i].cpu()))
#                 probabilities.append(probs[i].cpu().numpy())
#     return hard_negatives, probabilities

# # Track total training time
# start_time = time.time()

# # Train the model
# print("Starting training...")
# model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100)

# # Calculate total training time
# end_time = time.time()
# total_time = end_time - start_time
# training_time_minutes = total_time / 60

# # Save the best model from initial training
# model_dir = workspace_dir / "model"
# os.makedirs(model_dir, exist_ok=True)
# torch.save(model.state_dict(), model_dir / "best_model.pth")

# # Log training time
# with open(model_dir / "training_time.txt", "w") as f:
#     f.write(f"Total training time: {training_time_minutes:.2f} minutes")

# # Plotting loss and accuracy
# epochs = range(1, len(train_losses) + 1)

# plt.figure()
# plt.plot(epochs, train_losses, 'bo', label='Training loss')
# plt.plot(epochs, val_losses, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.savefig(model_dir / 'loss.png')

# plt.figure()
# plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.savefig(model_dir / 'accuracy.png')

# print(f"Total training time: {training_time_minutes:.2f} minutes")
# print("Initial training complete and model saved.")

# # Identify hard negatives in the validation set
# print("Identifying hard negatives in the validation set.")
# val_hard_negatives, val_probabilities = find_hard_negatives(model, val_loader)

# # Identify hard negatives in the sampled patches
# print("Identifying hard negatives in the sampled patches.")
# sampled_loader = DataLoader(sampled_dataset, batch_size=32, shuffle=False, num_workers=4)
# sampled_hard_negatives, sampled_probabilities = find_hard_negatives(model, sampled_loader)

# # Combine initial training data with hard negatives
# hard_negatives_dataset = [(image, label) for (image, label) in val_hard_negatives + sampled_hard_negatives]
# hard_negatives_loader = DataLoader(hard_negatives_dataset, batch_size=32, shuffle=True, num_workers=4)

# # Fine-tune the model with hard negatives
# print("Fine-tuning the model with hard negatives.")

# # Reload the best weights from initial training
# model.load_state_dict(best_model_wts)

# # Redefine optimizer and scheduler for fine-tuning
# fine_tune_optimizer = optim.Adam(model.parameters(), lr=0.0001)
# fine_tune_scheduler = ExponentialLR(fine_tune_optimizer, gamma=0.97)

# # Fine-tune the model
# model = train_model(model, hard_negatives_loader, val_loader, criterion, fine_tune_optimizer, fine_tune_scheduler, num_epochs=50)

# # Save the fine-tuned model
# torch.save(model.state_dict(), model_dir / "fine_tuned_model.pth")

# print("Fine-tuning complete and fine-tuned model saved.")
