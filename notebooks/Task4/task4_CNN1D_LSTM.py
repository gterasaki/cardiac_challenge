##---------------------------- Import Packages ----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from models import *
from torch.nn import DataParallel

##---------------------------- Load Data ----------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
print(f"Using device: {device}")
torch.set_default_dtype(torch.float64)

path = './../intracardiac_dataset/'
train_test_ratio = 0.8
VmTrainData, pECGTrainData, VmTestData, pECGTestData, actTimeTrain, actTimeTest  = fileReader(path, 16116, train_test_ratio)
print('Data loading from files - complete')

VmTrainData = (VmTrainData - torch.min(VmTrainData))/(torch.max(VmTrainData)-torch.min(VmTrainData))
pECGTrainData = (pECGTrainData - torch.min(pECGTrainData))/(torch.max(pECGTrainData) - torch.min(pECGTrainData))

# Normalize test data as well
VmTestData = (VmTestData - torch.min(VmTestData)) / (torch.max(VmTestData) - torch.min(VmTestData))
pECGTestData = (pECGTestData - torch.min(pECGTestData)) / (torch.max(pECGTestData) - torch.min(pECGTestData))

# Verify the shapes of your data
print(f'VmTrainData shape: {VmTrainData.shape}')
print(f'pECGTrainData shape: {pECGTrainData.shape}')

##---------------------------- Train/CV Split ----------------------------------------------------

# Convert tensors to numpy arrays for train_test_split
VmTrainData_np = VmTrainData.cpu().numpy()
pECGTrainData_np = pECGTrainData.cpu().numpy()

# Split the data into training and validation sets using train_test_split
VmTrainData_np, VmValData_np, pECGTrainData_np, pECGValData_np = train_test_split(
    VmTrainData_np, pECGTrainData_np, test_size=0.2, random_state=42)

# Convert back to tensors
VmTrainData = torch.tensor(VmTrainData_np).to(device)
pECGTrainData = torch.tensor(pECGTrainData_np).to(device)
VmValData = torch.tensor(VmValData_np).to(device)
pECGValData = torch.tensor(pECGValData_np).to(device)

# Create TensorDatasets
train_dataset = TensorDataset(VmTrainData, pECGTrainData)
val_dataset = TensorDataset(VmValData, pECGValData)

train_batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=train_batch_size, shuffle=False, generator=torch.Generator(device='cuda'))


print(f"Training data: {len(train_dataset)} samples")
print(f"Validation data: {len(val_dataset)} samples")

##---------------------------- Define Models ----------------------------------------------------

# Wrap the model in DataParallel to use multiple GPUs
model1 = SimpleCNNLSTM(input_channels=12, input_length=500).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model1 = DataParallel(model1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# Adjust the learning rate scheduler for more noticeable decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

##---------------------------- Model Parameter Numbers ------------------------------------------------

total_params_model1 = sum(p.numel() for p in model1.parameters())
trainable_params_model1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)

print(f'Model 1 - Total parameters: {total_params_model1}')
print(f'Model 1 - Trainable parameters: {trainable_params_model1}')

##---------------------------- Training Loop ----------------------------------------------------

# Initialize lists to store losses
train_losses = []
val_losses = []

EPOCHS = 4000 
train_interval = 400  
model_interval = 800  
pbar = tqdm(range(EPOCHS), desc='Training')

for epoch in pbar:
    model1.train()  
    running_loss = 0.0
    for i, (Vm, pECG) in enumerate(train_loader):
        Vm = Vm.to(device, non_blocking=True)
        pECG = pECG.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model1(pECG)
        loss = criterion(outputs, Vm)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate the average training loss for the epoch
    avg_train_loss = running_loss / (i + 1)
    train_losses.append(avg_train_loss)

    # Evaluate the model on the validation set
    model1.eval()  
    val_loss = 0.0
    with torch.no_grad():
        for Vm, pECG in val_loader:
            # Move validation data to the device
            Vm = Vm.to(device, non_blocking=True)
            pECG = pECG.to(device, non_blocking=True)

            # Model prediction
            outputs = model1(pECG)
            loss = criterion(outputs, Vm)
            val_loss += loss.item()

    # Calculate the average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if (epoch + 1) % train_interval == 0 or epoch == 0:
        pbar.set_description(f'Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')

    print(f"Done with Epoch {epoch + 1}")

    # Save the model periodically if required
    if (epoch + 1) % model_interval == 0:
        if not os.path.isdir('model_weights'):
            os.mkdir('model_weights')
        PATH = f'./model_weights/mycnn_lstm-epochs-{epoch+1}.pth'
        torch.save(model1.state_dict(), PATH)


