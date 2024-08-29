# ------------------ LOAD DEPENDENCIES ----------------------------------------------------
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Create directory for storing history and plots
os.makedirs('history', exist_ok=True)

# ------------------ LOAD Dataset ----------------------------------------------------
df_ptbd_normal = pd.read_csv("../ecg_dataset/ptbdb_normal.csv", header=None)
df_ptbd_abnormal = pd.read_csv("../ecg_dataset/ptbdb_abnormal.csv", header=None)

RANDOM_STATE = 0

# Split the normal and abnormal datasets into train, validation (CV), and test sets
X_train_normal, X__normal, y_train_normal, y__normal = train_test_split(
    df_ptbd_normal.iloc[:, :-1], df_ptbd_normal.iloc[:, -1], test_size=0.20, random_state=RANDOM_STATE)
X_train_abnormal, X__abnormal, y_train_abnormal, y__abnormal = train_test_split(
    df_ptbd_abnormal.iloc[:, :-1], df_ptbd_abnormal.iloc[:, -1], test_size=0.20, random_state=RANDOM_STATE)

X_cv_normal, X_test_normal, y_cv_normal, y_test_normal = train_test_split(
    X__normal, y__normal, test_size=0.5, random_state=RANDOM_STATE)
X_cv_abnormal, X_test_abnormal, y_cv_abnormal, y_test_abnormal = train_test_split(
    X__abnormal, y__abnormal, test_size=0.5, random_state=RANDOM_STATE)

# Determine the size for undersampling (number of samples in the smaller training class)
min_train_size = min(len(X_train_normal), len(X_train_abnormal))

# Perform undersampling on the training set
X_train_normal_under = X_train_normal.sample(n=min_train_size, random_state=42)
y_train_normal_under = y_train_normal.sample(n=min_train_size, random_state=42)

X_train_abnormal_under = X_train_abnormal.sample(n=min_train_size, random_state=42)
y_train_abnormal_under = y_train_abnormal.sample(n=min_train_size, random_state=42)

# Combine the undersampled training data
X_train = pd.concat([X_train_normal_under, X_train_abnormal_under], axis=0)
y_train = pd.concat([y_train_normal_under, y_train_abnormal_under], axis=0)

# Shuffle the combined training data
X_train, y_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True), y_train.sample(frac=1, random_state=42).reset_index(drop=True)
# Convert data to NumPy arrays
X_train_np = X_train.values
y_train_np = y_train.values

X_cv = np.concatenate((X_cv_normal, X_cv_abnormal))
y_cv = np.concatenate((y_cv_normal, y_cv_abnormal))

X_test = np.concatenate((X_test_normal, X_test_abnormal))
y_test = np.concatenate((y_test_normal, y_test_abnormal))

# Print the shapes of the combined datasets
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

print(f"Shape of X_cv: {X_cv.shape}")
print(f"Shape of y_cv: {y_cv.shape}")

print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

X_cv_tensor = torch.tensor(X_cv, dtype=torch.float32).unsqueeze(1)
y_cv_tensor = torch.tensor(y_cv, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training, validation, and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
cv_dataset = TensorDataset(X_cv_tensor, y_cv_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
cv_loader = DataLoader(cv_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ------------------ Build Model ----------------------------------------------------

class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3008, 512)  
        self.fc2 = nn.Linear(512, 216)
        self.fc3 = nn.Linear(216, 1)
        self.leaky_relu = nn.LeakyReLU(0.001)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Model, criterion, and optimizer
model = CNN1DModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------ Train Model ----------------------------------------------------

num_epochs = 100
history = {'train_loss': [], 'cv_loss': []}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    history['train_loss'].append(train_loss)

    model.eval()
    cv_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in cv_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            cv_loss += loss.item() * X_batch.size(0)
    
    cv_loss /= len(cv_loader.dataset)
    history['cv_loss'].append(cv_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, CV Loss: {cv_loss:.4f}')

torch.save(model.state_dict(), 'history/task1_1Dcnn_task1.pth')

# ------------------ Evaluate Model ----------------------------------------------------

# Evaluate the CNN model on the test set
with torch.no_grad():
    y_test_pred_prob = model(X_test_tensor.to(device)).squeeze().cpu().numpy()
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# Calculate accuracy scores
test_accuracy = accuracy_score(y_test_tensor.numpy(), y_test_pred)

# Print the accuracy for  testset
print(f"Test Accuracy: {test_accuracy:.4f}")

# ------------------ Plotting ----------------------------------------------------

# Classification report and confusion matrix for the test set
print("\nCNN Model Evaluation\n")
print("Classification Report (Test):")
print(classification_report(y_test_tensor.numpy(), y_test_pred))

print("Confusion Matrix (Test):")
conf_test_cnn = confusion_matrix(y_test_tensor.numpy(), y_test_pred)
print(conf_test_cnn)

# Visualization for the Test Confusion Matrix
plt.figure(figsize=(10, 7.5))
sns.heatmap(conf_test_cnn, annot=True, cmap='BrBG', fmt='g')
plt.title("Confusion Matrix - Test Set (CNN)")
# Save the plot
plt.savefig('history/cnn_test_confusion_matrix.png')

# Show the plot
plt.show()
