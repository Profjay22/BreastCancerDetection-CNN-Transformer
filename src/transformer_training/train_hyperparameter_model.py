import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

class SlideDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.labels = [1 if 'tumor' in f else 0 for f in self.files]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        features = torch.from_numpy(data['features']).float()
        label = torch.tensor(self.labels[idx]).long()
        return features, label

class TransformerModel(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding = torch.nn.Linear(1024, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(d_model, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

def stratified_split(dataset, val_size=0.2, val_normal_ratio=0.6):
    normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    tumor_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    
    n_val = int(len(dataset) * val_size)
    n_val_normal = int(n_val * val_normal_ratio)
    n_val_tumor = n_val - n_val_normal
    
    val_normal = np.random.choice(normal_indices, n_val_normal, replace=False)
    val_tumor = np.random.choice(tumor_indices, n_val_tumor, replace=False)
    
    val_indices = np.concatenate([val_normal, val_tumor])
    train_indices = np.array(list(set(range(len(dataset))) - set(val_indices)))
    
    print(f"Total samples: {len(dataset)}")
    print(f"Training set size: {len(train_indices)} ({len(train_indices)/len(dataset):.2%})")
    print(f"Validation set size: {len(val_indices)} ({len(val_indices)/len(dataset):.2%})")
    print(f"Validation set composition: {len(val_normal)} normal ({len(val_normal)/len(val_indices):.2%}), "
          f"{len(val_tumor)} tumor ({len(val_tumor)/len(val_indices):.2%})")
    
    return train_indices, val_indices

# Load the best hyperparameters
with open('/data/ja235/camelyon16_project/hyperparameter_tuning/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

# Convert necessary parameters to integers
best_params['batch_size'] = int(best_params['batch_size'])
best_params['d_model'] = int(best_params['d_model'])
best_params['dim_feedforward'] = int(best_params['dim_feedforward'])
best_params['nhead'] = int(best_params['nhead'])
best_params['num_layers'] = int(best_params['num_layers'])

# Ensure d_model is divisible by nhead
best_params['d_model'] = max(64, (best_params['d_model'] // best_params['nhead']) * best_params['nhead'])

# Create the model with best parameters
model = TransformerModel(
    d_model=best_params['d_model'],
    nhead=best_params['nhead'],
    num_layers=best_params['num_layers'],
    dim_feedforward=best_params['dim_feedforward'],
    dropout=best_params['dropout']
)

# Load and split the data
data_dir = '/data/ja235/camelyon16_project/preprocessed_slides'
dataset = SlideDataset(data_dir)
train_indices, val_indices = stratified_split(dataset, val_size=0.2, val_normal_ratio=0.6)
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)

train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_data, batch_size=best_params['batch_size'])

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

# Training loop
num_epochs = 100  # Adjust as needed
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_accuracy = 0
best_model_state = None

for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_accuracy = train_correct / train_total
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_predictions = []
    val_true_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_predictions.extend(predicted.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
    
    val_accuracy = val_correct / val_total
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()

# Create a directory to save results
results_dir = '/data/ja235/camelyon16_project/training_results'
os.makedirs(results_dir, exist_ok=True)

# Plot and save training vs validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))
plt.close()

# Plot and save training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
plt.close()

# Compute and plot confusion matrix
cm = confusion_matrix(val_true_labels, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# Compute and plot ROC curve
fpr, tpr, _ = roc_curve(val_true_labels, val_predictions)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
plt.close()

# Save the best model
torch.save(best_model_state, os.path.join(results_dir, 'best_model.pth'))

print(f"Training completed. Results and best model saved in {results_dir}")