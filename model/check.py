# import os
# import time
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.optim.lr_scheduler import ExponentialLR
# from torchvision.datasets import ImageFolder
# from torchvision.transforms import Compose, ToTensor, Normalize
# from torch.utils.data import DataLoader
# from torchvision.models import googlenet, GoogLeNet_Weights
# from pathlib import Path
# from pytorch_lightning import Trainer, LightningModule
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# import psutil
# from torch.profiler import profile, ProfilerActivity

# # Define data paths
# workspace_dir = Path("/data/ja235/camelyon16_project/")
# train_patches_dir = workspace_dir / "data/augmented_patches/train"
# val_patches_dir = workspace_dir / "data/augmented_patches/val"

# # Set float32 matmul precision for Tensor Cores utilization
# torch.set_float32_matmul_precision('high')

# # Define transformations
# transform = Compose([
#     ToTensor(),
#     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# class GoogleNetModel(LightningModule):
#     def __init__(self):
#         super(GoogleNetModel, self).__init__()
#         weights = GoogLeNet_Weights.DEFAULT
#         self.model = googlenet(weights=weights)
#         self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Assuming 2 classes: normal and tumor
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         inputs, labels = batch
#         outputs = self.model(inputs)
#         loss = self.criterion(outputs, labels)
#         self.log('train_loss', loss, on_step=True, on_epoch=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         inputs, labels = batch
#         outputs = self.model(inputs)
#         loss = self.criterion(outputs, labels)
#         self.log('val_loss', loss, on_step=False, on_epoch=True)
#         self.log('val_acc', torch.sum(outputs.argmax(dim=1) == labels).float() / len(labels), on_step=False, on_epoch=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         scheduler = ExponentialLR(optimizer, gamma=0.97)
#         return [optimizer], [scheduler]

#     @staticmethod
#     def log_system_memory():
#         print(f"Current system memory usage: {psutil.virtual_memory().percent}%")

# def profile_batch(model, batch):
#     print("Profiling a batch...")
#     inputs, labels = batch
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
#                  record_shapes=True, 
#                  profile_memory=True, 
#                  with_stack=True) as prof:
#         outputs = model(inputs)
#         loss = model.criterion(outputs, labels)
#         loss.backward()
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#     GoogleNetModel.log_system_memory()
#     print("Batch profiling complete.")

# def main():
#     print("Starting the data loading process...")
#     start_time = time.time()

#     # Load datasets
#     train_dataset = ImageFolder(train_patches_dir, transform=transform)
#     val_dataset = ImageFolder(val_patches_dir, transform=transform)
#     print("Data loaded successfully.")

#     # Data loaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
#     print("Data loaders created.")

#     # Logger and callbacks
#     logger = TensorBoardLogger(save_dir=os.path.join(workspace_dir, 'logs'), name='GoogleNetModel')
#     checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(workspace_dir, 'model'), save_top_k=1, monitor='val_loss', mode='min', every_n_epochs=10)
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')

#     # Trainer
#     trainer = Trainer(
#         max_epochs=100,
#         accelerator='gpu',
#         devices=1,
#         precision='16-mixed',  # Optimizes memory and compute speed
#         callbacks=[checkpoint_callback, early_stopping],
#         logger=logger
#     )

#     # Model
#     model = GoogleNetModel()
#     print("Starting training...")

#     # Profile the first batch
#     for batch in train_loader:
#         print("Profiling first batch...")
#         profile_batch(model, batch)
#         break
    
#     trainer.fit(model, train_loader, val_loader)
#     print("Training complete.")

#     # Monitoring memory after training
#     GoogleNetModel.log_system_memory()

#     print(f"Training complete in {time.time() - start_time:.2f} seconds.")

# if __name__ == "__main__":
#     main()
