import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from bayes_opt import BayesianOptimization
import os
import json
from tqdm import tqdm

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
    
    n_val_normal = min(n_val_normal, len(normal_indices))
    n_val_tumor = min(n_val_tumor, len(tumor_indices))
    
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

def train_and_evaluate(d_model, nhead, num_layers, dim_feedforward, dropout, lr, batch_size, weight_decay):
    d_model = int(d_model)
    nhead = int(nhead)
    
    # Ensure d_model is divisible by nhead
    d_model = max(64, ((d_model // nhead) * nhead))
    
    model = TransformerModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=int(num_layers),
        dim_feedforward=int(dim_feedforward),
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_data, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=int(batch_size))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Gradient accumulation steps
    accumulation_steps = 4  # Adjust this value as needed
    
    for epoch in tqdm(range(30), desc="Epochs"):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            # Gradient scaling and accumulation
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

# Load the data
data_dir = '/data/ja235/camelyon16_project/preprocessed_slides'
dataset = SlideDataset(data_dir)

# Perform stratified split
train_indices, val_indices = stratified_split(dataset, val_size=0.2, val_normal_ratio=0.6)
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)

# Define the search space
pbounds = {
    'd_model': (64, 256),
    'nhead': (1, 8),  # Changed back to 8 as the maximum
    'num_layers': (1, 4),
    'dim_feedforward': (64, 1024),
    'dropout': (0.1, 0.5),
    'lr': (1e-5, 1e-2),
    'batch_size': (4, 32),
    'weight_decay': (1e-5, 1e-2)
}

optimizer = BayesianOptimization(
    f=train_and_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Perform optimization
best_accuracy = 0
for i in tqdm(range(100), desc="Bayesian Optimization Iterations"):
    optimizer.maximize(init_points=1 if i == 0 else 0, n_iter=1)
    
    # Print current best
    if optimizer.max['target'] > best_accuracy:
        best_accuracy = optimizer.max['target']
        print(f"\nNew best accuracy: {best_accuracy:.4f}")
        print("Parameters:")
        for key, value in optimizer.max['params'].items():
            print(f"{key}: {value}")
        print()

# Print the best parameters
print("\nBest parameters found:")
print(optimizer.max['params'])
print(f"Best accuracy: {optimizer.max['target']:.4f}")

# Save the best parameters
save_dir = '/data/ja235/camelyon16_project/hyperparameter_tuning'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'best_hyperparameters.json')

with open(save_path, 'w') as f:
    json.dump(optimizer.max['params'], f)

print(f"Best hyperparameters saved to: {save_path}")

# Load the best parameters and create the final model
with open(save_path, 'r') as f:
    best_params = json.load(f)

best_model = TransformerModel(
    d_model=max(64, (int(best_params['d_model']) // int(best_params['nhead'])) * int(best_params['nhead'])),
    nhead=int(best_params['nhead']),
    num_layers=int(best_params['num_layers']),
    dim_feedforward=int(best_params['dim_feedforward']),
    dropout=best_params['dropout']
)

print("Best model created with the following parameters:")
print(best_params)