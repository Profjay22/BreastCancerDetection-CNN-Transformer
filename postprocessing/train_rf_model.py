import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import joblib
import os
import time
from tqdm import tqdm

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

# Perform initial stratified split (80% train, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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

# Define the hyperparameter search space
search_spaces = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(5, 100),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'bootstrap': Categorical([True, False]),
    'class_weight': Categorical([None, 'balanced', 'balanced_subsample'])
}

# Perform Bayesian optimization for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
bayes_search = BayesSearchCV(
    rf,
    search_spaces,
    n_iter=100,  # Increase this for more extensive search
    cv=5,
    n_jobs=-1,
    random_state=42,
    scoring='roc_auc'
)

# Integrate tqdm with the fit method to show the progress bar
with tqdm(total=bayes_search.total_iterations) as pbar:
    def on_step(optim_result):
        pbar.update(1)
        return True

    bayes_search.fit(X_train_scaled, y_train, callback=on_step)

# Get the best model
best_rf = bayes_search.best_estimator_

# Train the model and get training scores
train_scores = best_rf.fit(X_train_scaled, y_train).predict_proba(X_train_scaled)[:, 1]

# Make predictions on validation set
val_pred = best_rf.predict(X_val_scaled)
val_scores = best_rf.predict_proba(X_val_scaled)[:, 1]

# Calculate metrics
conf_matrix = confusion_matrix(y_val, val_pred)
accuracy = accuracy_score(y_val, val_pred)
precision = precision_score(y_val, val_pred)
recall = recall_score(y_val, val_pred)
f1 = f1_score(y_val, val_pred)
fpr, tpr, _ = roc_curve(y_val, val_scores)
roc_auc = auc(fpr, tpr)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
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
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Plot feature importance
feature_importance = best_rf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)
top_6_idx = sorted_idx[-6:]

plt.figure(figsize=(10, 6))
plt.barh(range(6), feature_importance[top_6_idx])
plt.yticks(range(6), [feature_names[i] for i in top_6_idx])
plt.xlabel('Feature Importance')
plt.title('Top 6 Important Features')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# Plot train vs validation loss
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5, scoring='neg_log_loss')
train_loss = -np.mean(cv_scores)
val_loss = -np.mean(np.log(val_scores + 1e-10) * y_val + np.log(1 - val_scores + 1e-10) * (1 - y_val))

plt.figure(figsize=(8, 6))
plt.bar(['Train', 'Validation'], [train_loss, val_loss])
plt.ylabel('Log Loss')
plt.title('Train vs Validation Loss')
plt.savefig(os.path.join(output_dir, 'train_val_loss.png'))
plt.close()

# Calculate total training time
total_time = time.time() - start_time

# Save metrics
with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Train Loss: {train_loss:.4f}\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Total Training Time: {total_time:.2f} seconds\n")
    f.write(f"Best hyperparameters: {bayes_search.best_params_}\n")

# Save the best model
joblib.dump(best_rf, os.path.join(output_dir, 'best_random_forest_model.joblib'))

# Save the scaler
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))

# Save the imputer
joblib.dump(imputer, os.path.join(output_dir, 'imputer.joblib'))

print("Model training and evaluation completed. Results saved in:", output_dir)
