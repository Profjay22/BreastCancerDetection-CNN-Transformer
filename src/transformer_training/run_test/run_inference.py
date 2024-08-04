import os
import torch
from dataset import SlideDataset, collate_fn
from model import TransformerForSlideClassification
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the model
print("Loading model...")
model = TransformerForSlideClassification()
model.load_state_dict(torch.load('/data/ja235/camelyon16_project/src/transformer_training/best_model.pth'))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

# Prepare the test dataset
print("Preparing test dataset...")
normal_dir = '/data/ja235/camelyon16_project/test_extracted_features/normal'
tumor_dir = '/data/ja235/camelyon16_project/test_extracted_features/tumor'

test_dataset = SlideDataset(normal_dir, tumor_dir, max_patches_per_slide=1000)
print(f"Total slides in test dataset: {len(test_dataset)}")

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
print("Test DataLoader created.")

# Run inference and collect predictions
print("Running inference...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []
all_slide_keys = []

with torch.no_grad():
    for i, (features, labels, slide_keys) in enumerate(tqdm(test_loader, desc="Inference Progress")):
        print(f"Processing batch {i+1}/{len(test_loader)}")
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_slide_keys.extend(slide_keys)

# Calculate metrics
print("Calculating metrics...")
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_preds)
fpr, tpr, _ = roc_curve(all_labels, all_preds)

# Create directory for saving metrics
metrics_dir = '/data/ja235/camelyon16_project/src/transformer_training/metrics'
os.makedirs(metrics_dir, exist_ok=True)

# Save metrics to a text file
metrics_file = os.path.join(metrics_dir, 'metrics.txt')
with open(metrics_file, 'w') as f:
    f.write(f"Metric\t\tValue\n")
    f.write(f"Accuracy\t\t{accuracy:.4f}\n")
    f.write(f"Precision\t\t{precision:.4f}\n")
    f.write(f"Recall\t\t{recall:.4f}\n")
    f.write(f"F1 Score\t\t{f1:.4f}\n\n")

    report = classification_report(all_labels, all_preds, target_names=['Benign (class 0)', 'Malignant (class 1)'], output_dict=True)
    f.write(f"Class\t\tPrecision\tRecall\tF1-Score\tSupport\n")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            f.write(f"{class_name}\t\t{metrics['precision']:.4f}\t\t{metrics['recall']:.4f}\t\t{metrics['f1-score']:.4f}\t\t{metrics['support']}\n")
    f.write(f"Macro Avg\t\t{report['macro avg']['precision']:.4f}\t\t{report['macro avg']['recall']:.4f}\t\t{report['macro avg']['f1-score']:.4f}\t\t{report['macro avg']['support']}\n")
    f.write(f"Weighted Avg\t\t{report['weighted avg']['precision']:.4f}\t\t{report['weighted avg']['recall']:.4f}\t\t{report['weighted avg']['f1-score']:.4f}\t\t{report['weighted avg']['support']}\n")

print(f"Metrics saved to {metrics_file}")

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
conf_matrix_file = os.path.join(metrics_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_file)
plt.show()
print(f"Confusion matrix saved to {conf_matrix_file}")

# Plot ROC curve
plt.figure(figsize=(10,7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
roc_curve_file = os.path.join(metrics_dir, 'roc_curve.png')
plt.savefig(roc_curve_file)
plt.show()
print(f"ROC curve saved to {roc_curve_file}")

# Save the accuracy plot
epochs = list(range(1, 2))  # Inference is only one step, hence only one epoch
accuracy_values = [accuracy]

plt.figure(figsize=(10, 7))
plt.plot(epochs, accuracy_values, marker='o', label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.legend(loc="lower right")
accuracy_plot_file = os.path.join(metrics_dir, 'accuracy_plot.png')
plt.savefig(accuracy_plot_file)
plt.show()
print(f"Accuracy plot saved to {accuracy_plot_file}")

# Save the predictions
predictions = {
    "slide_keys": all_slide_keys,
    "predictions": all_preds,
    "labels": all_labels
}

predictions_file = os.path.join(metrics_dir, 'test_predictions.npz')
np.savez(predictions_file, **predictions)
print(f"Predictions saved to {predictions_file}")
