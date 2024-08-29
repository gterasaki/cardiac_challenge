import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time  # Import the time module

# Define the shapes
total_samples_train = 80390
flat_feature_size = 224 * 224

# Load the memory-mapped arrays
X_train_flat = np.memmap('./gaf_task2/X_train_flat_memmap.dat', dtype='float32', mode='r+', shape=(total_samples_train, flat_feature_size))
y_train = np.memmap('./gaf_task2/y_train_memmap.dat', dtype='int32', mode='r+', shape=(total_samples_train,))   

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

# Reshape the flattened arrays back to 2D images
image_size = (224, 224)
X_train = X_train_flat.reshape(-1, 1, *image_size)
X_cv = X_cv_flat.reshape(-1, 1, *image_size)
X_test = X_test_flat.reshape(-1, 1, *image_size)

# Convert labels to tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  
y_cv_tensor = torch.tensor(y_cv, dtype=torch.long)  
y_test_tensor = torch.tensor(y_test, dtype=torch.long)   

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_cv_tensor = torch.tensor(X_cv, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoader for training, validation, and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
cv_dataset = TensorDataset(X_cv_tensor, y_cv_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
cv_loader = DataLoader(cv_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

# Check the unique values in the labels
print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_cv:", np.unique(y_cv))
print("Unique values in y_test:", np.unique(y_test))

# Define the CNN Model for 2D Images 
class task2_CNN2DModel(nn.Module):
    def __init__(self, num_classes=5):
        super(task2_CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(200704, 512)
        self.fc2 = nn.Linear(512, 216)
        self.fc3 = nn.Linear(216, num_classes)   
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
        x = self.fc3(x)   
        return x

# Initialize the model with 5 classes
model = task2_CNN2DModel(num_classes=5)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model, criterion, and optimizer setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize history dictionary to store losses
history = {'train_loss': [], 'cv_loss': []}

# Record the start time
start_time = time.time()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs} (Training)')
    for batch_idx, (inputs, targets) in pbar_train:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        pbar_train.set_postfix({'Batch Loss': loss.item()})
    
    train_loss /= len(train_loader.dataset)
    history['train_loss'].append(train_loss)

    # Validation (CV) phase
    model.eval()
    cv_loss = 0.0
    pbar_cv = tqdm(enumerate(cv_loader), total=len(cv_loader), desc=f'Epoch {epoch+1}/{num_epochs} (Validation)')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar_cv:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            cv_loss += loss.item() * inputs.size(0)
            pbar_cv.set_postfix({'Batch Loss': loss.item()})
    
    cv_loss /= len(cv_loader.dataset)
    history['cv_loss'].append(cv_loss)

    # Print training and validation loss
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, CV Loss: {cv_loss:.4f}')

# Record the end time
end_time = time.time()

# Calculate and print the total training time
training_time = end_time - start_time
print(f"Total Training Time: {training_time:.2f} seconds")

# Create directory to save history and model
os.makedirs('history', exist_ok=True)

# Save the model
torch.save(model.state_dict(), 'history/task2_2Dcnn.pth')

# Save history as a CSV file
history_df = pd.DataFrame(history)
history_df.to_csv('history/history_task2_2DCNN.csv', index=False)

# Evaluation function to handle validation and test sets
def evaluate_model(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    pbar_eval = tqdm(loader, desc='Evaluating')   
    with torch.no_grad():
        for inputs, targets in pbar_eval:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

# Evaluate the model using the test_loader
y_test_pred, y_test_true = evaluate_model(test_loader, model, device)
test_accuracy = accuracy_score(y_test_true, y_test_pred)

# Print the accuracy for the test set
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report and confusion matrix for the test set
print("\nCNN Model Evaluation\n")
print("Classification Report (Test):")
print(classification_report(y_test_true, y_test_pred, zero_division=0))

print("Confusion Matrix (Test):")
conf_test_cnn = confusion_matrix(y_test_true, y_test_pred)
print(conf_test_cnn)

# Visualization for the Test Confusion Matrix
plt.figure(figsize=(10, 7.5))
sns.heatmap(conf_test_cnn, annot=True, cmap='BrBG', fmt='g')
plt.title("Confusion Matrix - Test Set (CNN)")
plt.savefig('history/task2_2Dcnn_test_confusion_matrix.png')
plt.show()
