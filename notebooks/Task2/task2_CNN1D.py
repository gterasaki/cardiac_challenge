import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Import the time module

# Load the datasets
df_mitbih_train = pd.read_csv("../ecg_dataset/mitbih_train.csv", header=None)
df_mitbih_test = pd.read_csv("../ecg_dataset/mitbih_test.csv", header=None)

# Print shapes of the dataframes
print("The shape of the mitbih_train is:", df_mitbih_train.shape)
print("The shape of the mitbih_test is:", df_mitbih_test.shape)

# Combine the datasets
df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)
df_mitbih.rename(columns={187: 'label'}, inplace=True)

# Print shape of the combined dataframe
print("The shape of the combined mitbih dataframe is:", df_mitbih.shape)

# Split data into features and labels
X = df_mitbih.iloc[:, :-1]  # Features
y = df_mitbih['label']       # Labels

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)

# Resample the training data
target_samples = 8039 * 2  # Desired number of samples per class

def resample_class(X, y, target_samples, is_majority=False):
    return resample(X, y, replace=not is_majority, n_samples=target_samples, random_state=42)

X_train_resampled = pd.DataFrame()
y_train_resampled = pd.Series(dtype='int')

for label in y_train.unique():
    X_class = X_train[y_train == label]
    y_class = y_train[y_train == label]
    
    if label == 0:  # Undersample the majority class
        X_res, y_res = resample_class(X_class, y_class, target_samples, is_majority=True)
    else:  # Oversample the minority classes
        X_res, y_res = resample_class(X_class, y_class, target_samples, is_majority=False)
    
    X_train_resampled = pd.concat([X_train_resampled, pd.DataFrame(X_res)], axis=0)
    y_train_resampled = pd.concat([y_train_resampled, pd.Series(y_res)], axis=0)

# Shuffle the resampled training set
X_train_resampled = X_train_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
y_train_resampled = y_train_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert the resampled training data back to NumPy arrays
X_train_np = X_train_resampled.values
y_train_np = y_train_resampled.values

# Convert validation and test sets to NumPy arrays
X_cv_np = X_cv
y_cv_np = y_cv.values

X_test_np = X_test
y_test_np = y_test.values

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)  
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

X_cv_tensor = torch.tensor(X_cv_np, dtype=torch.float32).unsqueeze(1)  
y_cv_tensor = torch.tensor(y_cv_np, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1)  
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

# Print shapes to verify
print(f"Shape of X_train_tensor: {X_train_tensor.shape}")
print(f"Shape of y_train_tensor: {y_train_tensor.shape}")
print(f"Shape of X_cv_tensor: {X_cv_tensor.shape}")
print(f"Shape of y_cv_tensor: {y_cv_tensor.shape}")
print(f"Shape of X_test_tensor: {X_test_tensor.shape}")
print(f"Shape of y_test_tensor: {y_test_tensor.shape}")

# Create DataLoader for training, validation, and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
cv_dataset = TensorDataset(X_cv_tensor, y_cv_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
cv_loader = DataLoader(cv_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class CNN1DModel_task2(nn.Module):
    def __init__(self, num_classes):
        super(CNN1DModel_task2, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.3)  # Adjusted dropout rate
        self.fc1 = nn.Linear(3008, 512)
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

# Define the number of classes in your dataset
num_classes = 5   

# Instantiate the model
model = CNN1DModel_task2(num_classes=num_classes)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create directory for storing history and plots
os.makedirs('history', exist_ok=True)

# Training loop
num_epochs = 100
history = {'train_loss': [], 'cv_loss': []}

# Start the timer
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()   
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
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
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()   
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            cv_loss += loss.item() * X_batch.size(0)
    
    cv_loss /= len(cv_loader.dataset)
    history['cv_loss'].append(cv_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, CV Loss: {cv_loss:.4f}')

# End the timer and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken to train TASK2 1DCNN the code: {elapsed_time:.2f} seconds")

# Save the model
torch.save(model.state_dict(), 'history/task2_1Dcnn.pth')

# Evaluate the CNN model on the training set
model.eval()
with torch.no_grad():
    y_train_pred_prob = model(X_train_tensor.to(device)).cpu().numpy()
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)

# Evaluate the CNN model on the CV set
with torch.no_grad():
    y_cv_pred_prob = model(X_cv_tensor.to(device)).cpu().numpy()
    y_cv_pred = np.argmax(y_cv_pred_prob, axis=1)

# Evaluate the CNN model on the test set
with torch.no_grad():
    y_test_pred_prob = model(X_test_tensor.to(device)).cpu().numpy()
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    
    
# Calculate accuracy scores
train_accuracy = accuracy_score(y_train_tensor.numpy(), y_train_pred)
cv_accuracy = accuracy_score(y_cv_tensor.numpy(), y_cv_pred)
test_accuracy = accuracy_score(y_test_tensor.numpy(), y_test_pred)

# Print the accuracy for each dataset
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {cv_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report and confusion matrix for the test set
print("\nCNN Model Evaluation\n")
print("Classification Report (Test):")
print(classification_report(y_test_tensor.numpy(), y_test_pred))

print("Confusion Matrix (Test):")
conf_test_cnn = confusion_matrix(y_test_tensor.numpy(), y_test_pred)
print(conf_test_cnn)

# Visualization for the Test Confusion Matrix
plt.figure(figsize=(10, 7.5))
sns.heatmap(conf_test_cnn, annot=True, cmap='Blues', fmt='g')
plt.title("Confusion Matrix - Test Set (CNN)")
# Save the plot
plt.savefig('history/task2_1Dcnn_test_confusion_matrix.png')

# Show the plot
plt.show()
