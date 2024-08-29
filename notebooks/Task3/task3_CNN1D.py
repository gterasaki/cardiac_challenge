import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import time  # Import time module

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your data
X_load = pd.read_csv('./gdata_task3/X_data_task3.csv')
X_data = np.array(X_load).reshape(16117, 500, 12)
y_load = pd.read_csv('./gdata_task3/Y_data_task3.csv')
Y_data = np.array(y_load).reshape(16117, 75, 1)

RANDOM_STATE = 0

# Normalize the dataset
X_normalized = np.zeros_like(X_data)
for i in range(12):
    for j in range(X_data.shape[0]):
        X_normalized[j, :, i] = (X_data[j, :, i]) / (np.max(X_data[j, :, i]) - np.min(X_data[j, :, i]))

Y_normalized = (Y_data - np.min(Y_data)) / (np.max(Y_data) - np.min(Y_data))

# Convert to PyTorch tensors and move to the appropriate device
X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32).to(device)

# Split into train, validation, and test sets
dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.8 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = int(val_test_size / 2)
test_size = val_test_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Print the shape of train, validation, and test datasets
for X_batch, y_batch in train_loader:
    print(f"Train batch shape: {X_batch.shape}, Labels batch shape: {y_batch.shape}")
    break  # Print shape of only one batch

for X_batch, y_batch in val_loader:
    print(f"Validation batch shape: {X_batch.shape}, Labels batch shape: {y_batch.shape}")
    break  # Print shape of only one batch

for X_batch, y_batch in test_loader:
    print(f"Test batch shape: {X_batch.shape}, Labels batch shape: {y_batch.shape}")
    break  # Print shape of only one batch

# Print the total number of samples in each set
print(f"Total samples in train dataset: {len(train_dataset)}")
print(f"Total samples in validation dataset: {len(val_dataset)}")
print(f"Total samples in test dataset: {len(test_dataset)}")


class MyCNNModel(nn.Module):
    def __init__(self):
        super(MyCNNModel, self).__init__()
        # Reduce the number of filters
        self.conv1 = nn.Conv1d(12, 128, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(64)
        
        # Increase pooling to reduce dimensionality
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)  
        self.flatten = nn.Flatten()
        
        # Reduce the size of fully connected layers
        self.fc1 = nn.Linear(8000, 128)  
        self.fc2 = nn.Linear(128, 75)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.pool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)

        return x

model = MyCNNModel().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store loss history
train_losses = []
val_losses = []

# Record the start time of training
start_time = time.time()

# Training loop
epochs = 8000   

pbar = tqdm(range(epochs), desc='Training')  

for epoch in pbar:
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)   
        optimizer.zero_grad()
        outputs = model(inputs.transpose(1, 2))
        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)   
            outputs = model(inputs.transpose(1, 2))
            loss = criterion(outputs, targets.squeeze())
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Update progress bar description
    pbar.set_description(f'Epoch [{epoch+1}/{epochs}] - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Record the end time of training
end_time = time.time()

# Calculate and print the total training time
training_time = end_time - start_time
print(f"Total Training Time: {training_time:.2f} seconds")

# Save the model
torch.save(model.state_dict(), 'task3_my_cnn_model.pth')

# Save the loss history as a DataFrame
history_df = pd.DataFrame({
    'train_loss': train_losses,
    'val_loss': val_losses
})

# Save the history to a CSV file for future analysis
history_df.to_csv('training_history.csv', index=False)

# Test and evaluate the model
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
predictions = []
targets_list = []
with torch.no_grad():  
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 
        outputs = model(inputs.transpose(1, 2))   
        loss = criterion(outputs, targets.squeeze())   
        test_loss += loss.item()   
        predictions.append(outputs.cpu().numpy())   
        targets_list.append(targets.cpu().numpy())   

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# Concatenate predictions and targets for evaluation
predictions = np.concatenate(predictions, axis=0)
targets_concat = np.concatenate(targets_list, axis=0)

# Compute R2 Score
r2score = r2_score(targets_concat.reshape(predictions.shape), predictions)
print(f"R2 Score: {r2score:.4f}")

#---------------------------------------------
# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(history_df['train_loss'], label='Train Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the Activation Times
ActTime_real = Y_tensor[test_dataset.indices].cpu().numpy()[0].reshape(75, 1)  # Real Activation Time f 
ActTime_pred = predictions[0].reshape(75, 1)  # Predicted Activation Time 

# Determine the common color scale
vmin = min(ActTime_real.min(), ActTime_pred.min())
vmax = max(ActTime_real.max(), ActTime_pred.max())

# Plot the Real Activation Time
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ActTime_real, cmap='jet', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
plt.title('Real Activation Time')
plt.colorbar()
plt.xlabel('Time Step')
plt.ylabel('Activation Value')

# Plot the Predicted Activation Time
plt.subplot(1, 2, 2)
plt.imshow(ActTime_pred, cmap='jet', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
plt.title('Predicted Activation Time')
plt.colorbar()
plt.xlabel('Time Step')
plt.ylabel('Activation Value')

plt.tight_layout()
plt.show()

