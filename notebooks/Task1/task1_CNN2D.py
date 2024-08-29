# ------------------ LOAD DEPENDENCIES ----------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

# Set random state for reproducibility
RANDOM_STATE = 0

# ------------------ LOAD Dataset ----------------------------------------------------

# Load the flattened arrays and labels from .npy files with 'task1_' prefix
X_train_flat = np.load('./task1_X_train_flat.npy')
X_cv_flat = np.load('./task1_X_cv_flat.npy')
X_test_flat = np.load('./task1_X_test_flat.npy')
y_train = np.load('./task1_y_train.npy')
y_cv = np.load('./task1_y_cv.npy')
y_test = np.load('./task1_y_test.npy')

print("Flattened GAF images and labels have been loaded successfully with 'task1_' prefix.")

# Reshape the flattened arrays back to 2D images
image_size = (224, 224)  # Update this with the correct image size
X_train = X_train_flat.reshape(-1, 1, *image_size)
X_cv = X_cv_flat.reshape(-1, 1, *image_size)
X_test = X_test_flat.reshape(-1, 1, *image_size)

# Convert labels to tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_cv_tensor = torch.tensor(y_cv, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_cv_tensor = torch.tensor(X_cv, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoader for training, validation, and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
cv_dataset = TensorDataset(X_cv_tensor, y_cv_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
cv_loader = DataLoader(cv_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Print the shapes of the original flattened arrays
print("Shape of X_train_flat:", X_train_flat.shape)
print("Shape of X_cv_flat:", X_cv_flat.shape)
print("Shape of X_test_flat:", X_test_flat.shape)

# Print the shapes of the reshaped 2D arrays
print("Shape of X_train (reshaped to 2D):", X_train.shape)
print("Shape of X_cv (reshaped to 2D):", X_cv.shape)
print("Shape of X_test (reshaped to 2D):", X_test.shape)

# Print the shapes of the labels
print("Shape of y_train:", y_train.shape)
print("Shape of y_cv:", y_cv.shape)
print("Shape of y_test:", y_test.shape)

# Print the shapes of the tensors
print("Shape of X_train_tensor:", X_train_tensor.shape)
print("Shape of y_train_tensor:", y_train_tensor.shape)
print("Shape of X_cv_tensor:", X_cv_tensor.shape)
print("Shape of y_cv_tensor:", y_cv_tensor.shape)
print("Shape of X_test_tensor:", X_test_tensor.shape)
print("Shape of y_test_tensor:", y_test_tensor.shape)

# ------------------ Build Model ----------------------------------------------------

# Define a CNN Model for 2D Images
class task1_CNN2DModel(nn.Module):
    def __init__(self):
        super(task1_CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(401408, 512)
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

# Initialize the model
model = task1_CNN2DModel()

# Model, criterion, and optimizer setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------ Train Model ----------------------------------------------------

# Initialize history dictionary to store losses
history = {'train_loss': [], 'cv_loss': []}

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    history['train_loss'].append(train_loss)

    # Validation (CV) phase
    model.eval()
    cv_loss = 0.0
    with torch.no_grad():
        for inputs, targets in cv_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            cv_loss += loss.item() * inputs.size(0)
    
    cv_loss /= len(cv_loader.dataset)
    history['cv_loss'].append(cv_loss)

    # Print training and validation loss
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, CV Loss: {cv_loss:.4f}')

# Create directory to save history and model
os.makedirs('history', exist_ok=True)

# Save the model
torch.save(model.state_dict(), 'history/task1_2Dcnn.pth')

# Save history as a CSV file
history_df = pd.DataFrame(history)
history_df.to_csv('history/history_task1_2DCNN.csv', index=False)

# # Optionally, plot the training and validation loss curves
# plt.figure(figsize=(10, 5))
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['cv_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# ------------------ Evaluate Model ----------------------------------------------------

# Initialize list to store test set predictions
y_test_pred = []

# Evaluate on the test set
model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).squeeze().cpu().numpy()
        y_test_pred.extend((outputs > 0.5).astype(int))

# Convert predictions to numpy arrays
y_test_pred = np.array(y_test_pred)

# Calculate the test accuracy score
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the test accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report and confusion matrix for the test set
print("\nCNN Model Evaluation\n")
print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred, zero_division=1))

print("Confusion Matrix (Test):")
conf_test_cnn = confusion_matrix(y_test, y_test_pred)
print(conf_test_cnn)

# ------------------ Plotting ----------------------------------------------------

# Visualization for the Test Confusion Matrix
plt.figure(figsize=(10, 7.5))
sns.heatmap(conf_test_cnn, annot=True, cmap='BrBG', fmt='g')
plt.title("Confusion Matrix - Test Set (CNN)")
# Save the plot
plt.savefig('history/task1_2Dcnn_test_confusion_matrix.png')

# Show the plot
plt.show()
