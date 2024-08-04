import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to the required files
input_csv_path = "/data/ja235/camelyon16_project/test_results/test_slide_level_features_with_labels.csv"
model_path = "/data/ja235/camelyon16_project/model_results/best_random_forest_model.joblib"
output_dir = "/data/ja235/camelyon16_project/RF_test"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_csv(input_csv_path)

# Load the XGBoost model
model = joblib.load(model_path)

# Separate features and labels
X = df.drop(['slide_id', 'label'], axis=1)
y = df['label'].apply(lambda x: 1 if x == 'tumor' else 0)  # Convert labels to binary format

# Perform the classification
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred_proba)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
fpr, tpr, _ = roc_curve(y, y_pred_proba)

# Save metrics
with open(f"{output_dir}/classification_metrics.txt", 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(f"{output_dir}/roc_curve.png")
plt.close()

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='blue')
plt.ylim([0.0, 1.0])
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.savefig(f"{output_dir}/accuracy.png")
plt.close()

print("Classification metrics saved and results plotted.")
