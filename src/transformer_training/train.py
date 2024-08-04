import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, num_epochs=50, learning_rate=1e-4, patience=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.metrics = []
        self.writer = SummaryWriter()

    def train(self):
        start_time = time.time()
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_metrics = self._validate_one_epoch(epoch)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_metrics['accuracy'])
            self.metrics.append(val_metrics)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            self._log_epoch(epoch, train_loss, val_loss, train_acc, val_metrics)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss or val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_metrics['accuracy']
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Saved best model!")
                self.epochs_no_improve = 0
                self._plot_confusion_matrix(val_metrics['confusion_matrix'], epoch)
                self._plot_roc_curve(val_metrics['roc_curve'], val_metrics['roc_auc'], epoch)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve == self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break
            
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time/3600:.2f} hours")
        
        # Plot the training curves after all epochs
        self._plot_training_curves()
        self._save_metrics()
        self.writer.close()
        
        # Evaluate on test set if provided
        if self.test_loader is not None:
            self._evaluate_test_set()

    def _train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        for batch_idx, (features, labels, _) in enumerate(train_pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
        
        return train_loss / len(self.train_loader), correct / total

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Validation]")
        with torch.no_grad():
            for features, labels, _ in val_pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        val_metrics = self._calculate_metrics(all_labels, all_preds)
        return val_loss / len(self.val_loader), val_metrics

    def _evaluate_test_set(self):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels, _ in tqdm(self.test_loader, desc="Evaluating Test Set"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_metrics = self._calculate_metrics(all_labels, all_preds)
        print("\nTest Set Evaluation:")
        print(f"Test Loss: {test_loss/len(self.test_loader):.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        print(f"Test AUC-ROC: {test_metrics['roc_auc']:.4f}")
        
        self._plot_confusion_matrix(test_metrics['confusion_matrix'], 'test')
        self._plot_roc_curve(test_metrics['roc_curve'], test_metrics['roc_auc'], 'test')

    def _calculate_metrics(self, labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        cm = confusion_matrix(labels, preds)
        fpr, tpr, _ = roc_curve(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': auc,
            'mcc': mcc,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr)
        }

    def _log_epoch(self, epoch, train_loss, val_loss, train_acc, val_metrics):
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
        self.writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
        self.writer.add_scalar('AUC-ROC/Validation', val_metrics['roc_auc'], epoch)

    def _plot_confusion_matrix(self, cm, epoch):
        os.makedirs('confusion_matrices', exist_ok=True)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.savefig(f'confusion_matrices/confusion_matrix_epoch_{epoch}.png')
        plt.close()
        self.writer.add_figure(f'Confusion Matrix/Epoch {epoch}', plt.gcf())

    def _plot_roc_curve(self, roc_curve, roc_auc, epoch):
        os.makedirs('roc_curves', exist_ok=True)
        fpr, tpr = roc_curve
        plt.figure(figsize=(10,7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Epoch {epoch}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curves/roc_curve_epoch_{epoch}.png')
        plt.close()
        self.writer.add_figure(f'ROC Curve/Epoch {epoch}', plt.gcf())

    def _plot_training_curves(self):
        epochs = list(range(1, len(self.train_losses) + 1))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        
        os.makedirs('training_plots', exist_ok=True)
        
        plt.savefig(f'training_plots/training_curves_epoch_{len(epochs)}.png')
        plt.close()
        self.writer.add_figure('Training Curves', plt.gcf())

    def _save_metrics(self):
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv('training_metrics.csv', index=False)