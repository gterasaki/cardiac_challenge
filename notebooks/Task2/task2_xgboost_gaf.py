import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Define the shapes
total_samples_train = 80390
flat_feature_size = 224 * 224

# Load the memory-mapped arrays
X_train_flat = np.memmap('./gaf_task2/X_train_flat_memmap.dat', dtype='float32', mode='r+', shape=(total_samples_train, flat_feature_size))
y_train = np.memmap('./gaf_task2/y_train_memmap.dat', dtype='float32', mode='r+', shape=(total_samples_train,))

# Load X_test and y_test
X_test_flat = np.load('./gaf_task2/task2_X_test_flat.npy')
y_test = np.load('./gaf_task2/task2_y_test.npy')

# Load X_cv and y_cv
X_cv_flat = np.load('./gaf_task2/task2_X_cv_flat.npy')
y_cv = np.load('./gaf_task2/task2_y_cv.npy')

print("Data loaded successfully.")
print(f"Shape of X_train_flat_memmap: {X_train_flat.shape}")
print(f"Shape of y_train_memmap: {y_train.shape}")
print(f"Shape of X_test_flat: {X_test_flat.shape}")
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of X_cv_flat: {X_cv_flat.shape}")
print(f"Shape of y_cv: {y_cv.shape}")

RANDOM_STATE = 0

# Train XGBoost with GPU support
xgb_model2 = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4,
                           reg_alpha=0.1, reg_lambda=0.1, subsample=0.8,
                           colsample_bytree=0.8, verbosity=0, early_stopping_rounds=50, 
                           random_state=RANDOM_STATE, tree_method='gpu_hist', gpu_id=0)

xgb_model2.fit(X_train_flat, y_train,
               eval_set=[(X_cv_flat, y_cv)],
               verbose=True)

# Evaluate the model on the entire training, validation, and test sets
train_accuracy = accuracy_score(y_train, xgb_model2.predict(X_train_flat))
cv_accuracy = accuracy_score(y_cv, xgb_model2.predict(X_cv_flat))
test_accuracy = accuracy_score(y_test, xgb_model2.predict(X_test_flat))

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {cv_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
joblib.dump(xgb_model2, 'history/task2_2D_xgb_model500.pkl')

# XGBoost Model Evaluation
print("\nXGBOOST Evaluation\n")

# Classification Report on Test Data
print("Classification Report (Test):")
print(classification_report(y_test, xgb_model2.predict(X_test_flat)))

# Confusion Matrix on Test Data
print("Confusion Matrix (Test):")
conf_test = confusion_matrix(y_test, xgb_model2.predict(X_test_flat))
print(conf_test)

# Accuracy Score
print(f"Metrics test:\n\tAccuracy score: {accuracy_score(xgb_model2.predict(X_test_flat), y_test):.4f}")

# Visualization for the Test Confusion Matrix
plt.figure(figsize=(10, 7.5))
sns.heatmap(conf_test, annot=True, cmap='BrBG', fmt='g')
plt.title("Confusion Matrix - Test Set")
plt.savefig('history/task2_2D_xgb_confusion_matrix.png')  # Save plot
plt.show()

