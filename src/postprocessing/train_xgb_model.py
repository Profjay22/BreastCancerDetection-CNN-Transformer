import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import os
import time

# Start timing
start_time = time.time()

# Load the data
data_path = "/data/ja235/camelyon16_project/training_results_new/train_slide_level_features.csv"
df = pd.read_csv(data_path)

# Separate features and labels
X = df.drop(['slide_id', 'label'], axis=1)
y = df['label']

# Create output directory
output_dir = "/data/ja235/camelyon16_project/model_results"
os.makedirs(output_dir, exist_ok=True)

# Print overall class distribution
print(f"Overall class distribution: {y.value_counts(normalize=True)}")

# Perform initial stratified split (70% train, 30% temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split the temp set into normal and tumor
X_temp_normal = X_temp[y_temp == 0]
X_temp_tumor = X_temp[y_temp == 1]

# Calculate the desired numbers for the validation set
total_val_samples = len(X_temp)
desired_normal_samples = int(0.6 * total_val_samples)
desired_tumor_samples = total_val_samples - desired_normal_samples

# Sample from normal and tumor to create the validation set
X_val_normal = X_temp_normal.sample(n=desired_normal_samples, random_state=42)
X_val_tumor = X_temp_tumor.sample(n=desired_tumor_samples, replace=True, random_state=42)

# Combine to create the final validation set
X_val = pd.concat([X_val_normal, X_val_tumor])
y_val = pd.Series([0] * len(X_val_normal) + [1] * len(X_val_tumor), index=X_val.index)

# Add any unused samples back to the training set
X_train = pd.concat([X_train, X_temp_normal[~X_temp_normal.index.isin(X_val_normal.index)]])
X_train = pd.concat([X_train, X_temp_tumor[~X_temp_tumor.index.isin(X_val_tumor.index)]])
y_train = pd.concat([y_train, y_temp[~y_temp.index.isin(y_val.index)]])

# Ensure X_train and y_train have the same index
X_train = X_train.loc[y_train.index]

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Validation set composition: {sum(y_val == 0)} normal samples ({sum(y_val == 0)/len(y_val):.2%}), "
      f"{sum(y_val == 1)} tumor samples ({sum(y_val == 1)/len(y_val):.2%})")

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Hyperparameter tuning for XGBoost with Grid Search
xgb_output_dir = "/data/ja235/camelyon16_project/xgb_model_results"
os.makedirs(xgb_output_dir, exist_ok=True)

# Define the hyperparameter search space for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Perform Grid Search for hyperparameter tuning
xgb_model = xgb.XGBClassifier(random_state=42)
grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc',
    verbose=3  # Increased verbosity to show progress
)

grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_xgb = grid_search.best_estimator_

# Train the model and get training scores
train_scores_xgb = best_xgb.fit(X_train_scaled, y_train).predict_proba(X_train_scaled)[:, 1]

# Make predictions on validation set
val_pred_xgb = best_xgb.predict(X_val_scaled)
val_scores_xgb = best_xgb.predict_proba(X_val_scaled)[:, 1]

# Calculate metrics
conf_matrix_xgb = confusion_matrix(y_val, val_pred_xgb)
accuracy_xgb = accuracy_score(y_val, val_pred_xgb)
precision_xgb = precision_score(y_val, val_pred_xgb)
recall_xgb = recall_score(y_val, val_pred_xgb)
f1_xgb = f1_score(y_val, val_pred_xgb)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, val_scores_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(xgb_output_dir, 'confusion_matrix.png'))
plt.close()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(xgb_output_dir, 'roc_curve.png'))
plt.close()

# Plot feature importance
xgb.plot_importance(best_xgb, max_num_features=10)
plt.title('Top 10 Important Features')
plt.savefig(os.path.join(xgb_output_dir, 'feature_importance.png'))
plt.close()

# Plot train vs validation accuracy
train_acc_xgb = accuracy_score(y_train, best_xgb.predict(X_train_scaled))
val_acc_xgb = accuracy_score(y_val, val_pred_xgb)

plt.figure(figsize=(8, 6))
plt.bar(['Train', 'Validation'], [train_acc_xgb, val_acc_xgb])
plt.ylabel('Accuracy')
plt.title('XGBoost Train vs Validation Accuracy')
plt.savefig(os.path.join(xgb_output_dir, 'train_val_accuracy.png'))
plt.close()

# Plot train vs validation loss
cv_scores_xgb = cross_val_score(best_xgb, X_train_scaled, y_train, cv=5, scoring='neg_log_loss')
train_loss_xgb = -np.mean(cv_scores_xgb)
val_loss_xgb = -np.mean(np.log(val_scores_xgb + 1e-10) * y_val + np.log(1 - val_scores_xgb + 1e-10) * (1 - y_val))

plt.figure(figsize=(8, 6))
plt.bar(['Train', 'Validation'], [train_loss_xgb, val_loss_xgb])
plt.ylabel('Log Loss')
plt.title('XGBoost Train vs Validation Loss')
plt.savefig(os.path.join(xgb_output_dir, 'train_val_loss.png'))
plt.close()

# Calculate total training time
total_time_xgb = time.time() - start_time

# Save metrics
with open(os.path.join(xgb_output_dir, 'metrics.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy_xgb:.4f}\n")
    f.write(f"Precision: {precision_xgb:.4f}\n")
    f.write(f"Recall: {recall_xgb:.4f}\n")
    f.write(f"F1 Score: {f1_xgb:.4f}\n")
    f.write(f"ROC AUC: {roc_auc_xgb:.4f}\n")
    f.write(f"Train Loss: {train_loss_xgb:.4f}\n")
    f.write(f"Validation Loss: {val_loss_xgb:.4f}\n")
    f.write(f"Train Accuracy: {train_acc_xgb:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc_xgb:.4f}\n")
    f.write(f"Total Training Time: {total_time_xgb:.2f} seconds\n")
    f.write(f"Best hyperparameters: {grid_search.best_params_}\n")

# Save the best XGBoost model
joblib.dump(best_xgb, os.path.join(xgb_output_dir, 'best_xgboost_model.joblib'))

# Save the scaler
joblib.dump(scaler, os.path.join(xgb_output_dir, 'scaler.joblib'))

# Save the imputer
joblib.dump(imputer, os.path.join(xgb_output_dir, 'imputer.joblib'))

print("XGBoost model training and evaluation completed. Results saved in:", xgb_output_dir)
