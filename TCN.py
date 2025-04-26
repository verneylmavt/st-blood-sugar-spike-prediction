#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import joblib
from tqdm import tqdm
from IPython.display import Image, display

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchinfo import summary
from torchviz import make_dot
import shap
from captum.attr import IntegratedGradients

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# ## TCN

# ### Data Importing

# In[3]:


df = pd.read_csv("./.data/train_aug.csv")  # Load preprocessed training data


# In[4]:


print(df)  # Print the dataframe for inspection


# In[5]:


for col in df.columns:
    print(col)  # List all column names to verify dataset structure


# ### Data Splitting

# In[6]:


# Split data into training (80%) and temporary set (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
# Further split temporary set equally into validation and test sets (10% each overall)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


# In[7]:


print(f"Train Shape: {train_df.shape}")
print(f"Validation Shape: {val_df.shape}")
print(f"Test Shape: {test_df.shape}")


# In[8]:


# val_df.to_csv("./.data/train_val_aug.csv", index=False)
# test_df.to_csv("./.data/train_test_aug.csv", index=False)


# ### Dataset and DataLoader

# In[9]:


class BrisT1DDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train
        
        # If 'id' column exists, store it separately and remove it from features.
        if 'id' in self.df.columns:
            self.ids = self.df['id'].values
            self.df = self.df.drop(columns=['id'])
        else:
            self.ids = None
        
        # For training data, extract target column 'bg+1:00' and drop it from features.
        if self.is_train and 'bg+1:00' in self.df.columns:
            self.targets = self.df['bg+1:00'].values
            self.df = self.df.drop(columns=['bg+1:00'])
        else:
            self.targets = None
        
        # Convert remaining dataframe to a float32 numpy array.
        data = self.df.values.astype(np.float32)
        # Ensure the number of features is exactly 144 (6 features x 24 timesteps)
        if data.shape[1] != 144:
            raise ValueError(f"Expected 144, Real {data.shape[1]}")
        
        # Reshape the data to (num_samples, 6, 24) and then transpose it to (num_samples, 24, 6)
        # where 24 is the time dimension and 6 are the sensor features.
        self.X = data.reshape(-1, 6, 24).transpose(0, 2, 1)
        
        # Set target values if training; reshape targets to a column vector.
        if self.is_train:
            self.y = self.targets.astype(np.float32).reshape(-1, 1)
        else:
            self.y = None

    def __len__(self):
        return self.X.shape[0]  # Number of samples

    def __getitem__(self, idx):
        # Convert sample to torch tensor.
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.is_train:
            # Return both sample and corresponding target as torch tensor.
            target = torch.tensor(self.y[idx], dtype=torch.float32)
            return sample, target
        else:
            return sample


# In[10]:


# Instantiate datasets for training, validation, and testing.
train_dataset = BrisT1DDataset(train_df, is_train=True)
val_dataset = BrisT1DDataset(val_df, is_train=True)
test_dataset = BrisT1DDataset(test_df, is_train=True)


# In[11]:


batch_size = 128
# Create DataLoaders for batching data; training data is shuffled.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ### Model

# #### Mathematical Operation

# 1. **Input:** 
#    \begin{aligned} 
#    S_t^{(0)} \in \mathbb{R}^{6}, \quad t=1,\dots,24
#    \end{aligned}
# 
# 2. **Block 1 $( i=1 )$:**  
#    \begin{aligned}
#    d_1 &= 1,\quad p_1 = 1,\\[1mm]
#    S_t^{(1,1)} &= \mathrm{Dropout}\Bigl(\mathrm{ReLU}\Bigl(\text{chomp}\bigl(W_1^{(1)} \ast_{1} S_t^{(0)} + b_1^{(1)}\bigr)\Bigr)\Bigr),\\[1mm]
#    S_t^{(1,2)} &= \mathrm{Dropout}\Bigl(\mathrm{ReLU}\Bigl(\text{chomp}\bigl(W_2^{(1)} \ast_{1} S_t^{(1,1)} + b_2^{(1)}\bigr)\Bigr)\Bigr),\\[1mm]
#    \textbf{D}^{(1)} &: \mathbb{R}^{6} \to \mathbb{R}^{64},\\[1mm]
#    \tilde{S}_t^{(0)} &= 
#    \begin{cases}
#    S_t^{(0)} & \text{if } C_0 = C_1, \\
#    D^{(1)}\bigl(S_t^{(0)}\bigr) & \text{if } C_0 \neq C_1,
#    \end{cases}\\[2mm]
#    S_t^{(1,3)} &= \mathrm{ReLU}\Bigl( S_t^{(1,2)} + \tilde{S}_t^{(0)} \Bigr).
#    \end{aligned}
# 
# 3. **Block 2 $( i=2 )$:**  
#    \begin{aligned}
#    d_2 &= 2,\quad p_2 = 2,\\[1mm]
#    S_t^{(2,1)} &= \mathrm{Dropout}\Bigl(\mathrm{ReLU}\Bigl(\text{chomp}\bigl(W_1^{(2)} \ast_{2} S_t^{(1,3)} + b_1^{(2)}\bigr)\Bigr)\Bigr),\\[1mm]
#    S_t^{(2,2)} &= \mathrm{Dropout}\Bigl(\mathrm{ReLU}\Bigl(\text{chomp}\bigl(W_2^{(2)} \ast_{2} S_t^{(2,1)} + b_2^{(2)}\bigr)\Bigr)\Bigr),\\[1mm]
#    \textbf{D}^{(2)} &: \mathbb{R}^{64} \to \mathbb{R}^{32},\\[1mm]
#    \tilde{S}_t^{(1,3)} &= 
#    \begin{cases}
#    S_t^{(1,3)} & \text{if } C_1 = C_2, \\
#    D^{(2)}\bigl(S_t^{(1,3)}\bigr) & \text{if } C_1 \neq C_2,
#    \end{cases}\\[2mm]
#    S_t^{(2,3)} &= \mathrm{ReLU}\Bigl( S_t^{(2,2)} + \tilde{S}_t^{(1,3)} \Bigr).
#    \end{aligned}
# 
# 4. **Output:**  
#    Let $y = S_{24}^{(2,3)} \in \mathbb{R}^{32}$ be the feature vector at the final time step. Then,
#    \begin{aligned}
#    \hat{y} = U\, y + c, \quad U \in \mathbb{R}^{1 \times 32},\quad c \in \mathbb{R}.
#    \end{aligned}

# #### Computational Complexity

# 1. **Block 1:**  
#    \begin{aligned}
#    O\bigl(T \times (6 \times 64 \times 2 + 64 \times 64 \times 2 + 6 \times 64)\bigr)
#    \end{aligned}
# 
# 3. **Block 2:**  
#    \begin{aligned}
#    O\bigl(T \times (64 \times 32 \times 2 + 32 \times 32 \times 2 + 64 \times 32)\bigr)
#    \end{aligned}
# 
# 4. **Total:**  
#    For a batch of size $N$, the total complexity is:
#    \begin{aligned}
#    O\Bigl(N \times 24 \times \bigl(6 \times 64 \times 2 + 64 \times 64 \times 2 + 6 \times 64 + 64 \times 32 \times 2 + 32 \times 32 \times 2 + 64 \times 32\bigr)\Bigr)
#    \end{aligned}

# #### Model Architecture

# In[12]:


class BrisT1DChomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size  # Amount to trim from the end of the sequence
        
    def forward(self, x):
        # Trim the last 'chomp_size' elements along the temporal dimension
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


# In[13]:


class BrisT1DTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        # First convolutional layer with specified parameters.
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = BrisT1DChomp1d(padding)  # Remove excess padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer.
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                            stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = BrisT1DChomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers into a sequential module.
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        # If input and output channels differ, apply a 1x1 convolution to match dimensions for the residual connection.
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        # Residual connection: if downsample exists, adjust input; otherwise, use input as is.
        res = x if self.downsample is None else self.downsample(x)
        # Return the elementwise sum passed through ReLU.
        return self.relu(out + res)


# In[14]:


class BrisT1DTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.25):
        super().__init__()
        layers = []
        num_levels = len(num_channels)  # Number of temporal blocks to stack
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential increase in dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Calculate necessary padding to ensure output maintains temporal length before chomp.
            padding = (kernel_size - 1) * dilation_size
            layers.append(BrisT1DTemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                                dilation=dilation_size, padding=padding,
                                                dropout=dropout))
        # Sequentially stack all the temporal blocks.
        self.network = nn.Sequential(*layers)
        # Fully connected layer to map the final feature representation to a single output.
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # Transpose input from (batch, time, features) to (batch, features, time) for 1D convolution
        x = x.transpose(1, 2)
        y = self.network(x)
        # Use the last time step's output for prediction
        y = y[:, :, -1]
        out = self.fc(y)
        return out


# In[15]:


model = BrisT1DTCN(num_inputs=6, num_channels=[64, 32], kernel_size=2, dropout=0.25).to(device)
print(model)  # Print the model architecture


# #### Model Summary

# In[16]:


dummy_seq = torch.randn(1, 24, 6).to(device)  # Create a dummy input for visualization


# In[17]:


print("Model Summary:")
print(summary(model, input_data=[dummy_seq]))  # Print detailed summary including layer outputs and parameters


# #### Model Computational Graph

# In[18]:


dummy_output = model(dummy_seq)
dot = make_dot(dummy_output, params=dict(model.named_parameters()))  # Generate computation graph visualization


# In[19]:


print("Model Computational Graph:")
display(dot)  # Display the computational graph image


# ### Training

# In[ ]:


epochs = 100  # Maximum number of training epochs
criterion = nn.MSELoss()  # Loss function for regression
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer with specified learning rate
patience = 10  # Early stopping patience


# In[ ]:


train_losses = []  # List to store training loss for each epoch
val_losses = []    # List to store validation loss for each epoch
best_val_loss = float("inf")  # Initialize best validation loss to infinity
epochs_no_improve = 0  # Counter for early stopping


# In[ ]:


for epoch in range(epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        
        running_train_loss += loss.item() * X_batch.size(0)
    
    # Compute average training loss for the epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False):
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            running_val_loss += loss.item() * X_val.size(0)
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    
    # Check for improvement; save the best model and update early stopping counter.
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), './.export/checkpoint-aug-tcn.pt')
    else:
        epochs_no_improve += 1
        print(f"Early Stopping: {epochs_no_improve}/{patience}")
        
    if epochs_no_improve >= patience:
        print("Early Stopping!")
        break


# In[ ]:


plt.figure(figsize=(15, 10))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss vs Validation Loss")
plt.legend()
plt.show()  # Plot the loss curves to inspect training progress


# ### Evaluation

# In[20]:


# Load the best model saved during training
model.load_state_dict(torch.load('./.export/checkpoint-aug-tcn.pt'))


# In[21]:


# Define several evaluation metrics for regression performance
def rmse(pred, target):
    return np.sqrt(np.mean((pred - target)**2))

def mae(pred, target):
    return np.mean(np.abs(pred - target))

def mard(pred, target):
    return np.mean(np.abs(pred - target) / (np.abs(target) + 1e-6))

def mbe(pred, target):
    return np.mean(pred - target)

def mape(pred, target):
    return np.mean(np.abs((target - pred) / (target + 1e-6))) * 100

def pearson_r(pred, target):
    return np.corrcoef(pred, target)[0, 1]

def ccc(pred, target):
    pred_mean = np.mean(pred)
    target_mean = np.mean(target)
    pred_var = np.var(pred)
    target_var = np.var(target)
    covariance = np.mean((pred - pred_mean) * (target - target_mean))
    return (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean)**2 + 1e-6)


# In[22]:


model.eval()
preds = []
preds_targets = []

# Generate predictions on the test set
with torch.no_grad():
    for x_seq, targets in tqdm(test_loader, desc="Evaluation"):
        x_seq = x_seq.to(device)
        targets = targets.to(device)
        outputs = model(x_seq)
        preds.append(outputs.cpu().numpy())
        preds_targets.append(targets.cpu().numpy())
        
preds = np.concatenate(preds)
preds_targets = np.concatenate(preds_targets)


# In[23]:


# Load the target scaler to inverse transform scaled predictions and targets to original units
target_scaler = joblib.load('./.data/target_scaler_aug.pkl')
all_preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
all_targets = target_scaler.inverse_transform(preds_targets.reshape(-1, 1)).flatten()


# In[24]:


# Calibrate predictions using a simple Linear Regression model.
calibration_model = LinearRegression()
calibration_model.fit(all_preds.reshape(-1, 1), all_targets)
calibrated_val_preds = calibration_model.predict(all_preds.reshape(-1, 1))

all_preds = calibrated_val_preds


# In[25]:


residuals = all_preds - all_targets  # Compute residual errors


# In[26]:


plt.figure(figsize=(15, 10))
plt.scatter(all_targets, all_preds, alpha=0.4)
plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual Glucose vs Predicted Glucose")
plt.grid(True)
plt.show()  # Scatter plot of actual vs predicted values


# In[27]:


plt.figure(figsize=(15, 10))
plt.hist(residuals, bins=50)
plt.title("Residual")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()  # Histogram showing the distribution of residuals


# In[28]:


from sklearn.metrics import r2_score
test_r2 = r2_score(all_targets, all_preds)


# In[29]:


from sklearn.metrics import explained_variance_score
test_expl_var = explained_variance_score(all_targets, all_preds)


# In[30]:


test_rmse = rmse(all_preds, all_targets)
test_mae = mae(all_preds, all_targets)
test_mard = mard(all_preds, all_targets)

test_mbe = mbe(all_preds, all_targets)
test_r2 = r2_score(all_targets, all_preds)
test_expl_var = explained_variance_score(all_targets, all_preds)
test_mape = mape(all_preds, all_targets)
test_ccc = ccc(all_preds, all_targets)


# In[31]:


print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MSE: {test_rmse**2:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test MARD: {test_mard:.4f}")
print(f"Test MBE: {test_mbe:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test Explained Variance: {test_expl_var:.4f}")
print(f"Test MAPE: {test_mape:.2f}%")
print(f"Test CCC: {test_ccc:.4f}")


# ### Explanation

# In[32]:


# Reload the best model checkpoint for explanation analysis
model.load_state_dict(torch.load('./.export/checkpoint-aug-tcn.pt'))
model.eval()


# In[33]:


# Define human-readable names for each sensor feature.
sensor_names = ["bg (Blood Glucose Reading)", 
                "insulin (Insulin Intake)", 
                "carbs (Carbohydrate Intake)", 
                "hr (Mean Heart Rate)", 
                "steps (Steps Walked)", 
                "cals (Calories Burned)"]


# In[34]:


def avg_ig_values(model, test_loader=test_loader, n_samples=None):
    ig = IntegratedGradients(model)  # Initialize Integrated Gradients
    
    # Determine total number of samples in the test loader.
    total_samples = sum(x.shape[0] for x, _ in test_loader)
    if n_samples is None:
        n_samples = total_samples // 2  # Use half the samples by default for attribution analysis

    attr_accum = torch.zeros((24, 6), device=device)  # Initialize accumulation tensor for attributions
    delta_list = []  # To store convergence deltas from IG
    
    sampled = 0
    # Iterate over test batches
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        
        for i in range(x_batch.shape[0]):
            if sampled >= n_samples:
                break
            input_seq = x_batch[i:i+1]  # Process one sample at a time
            
            # Compute integrated gradients with a zero baseline and return convergence delta
            attr, delta = ig.attribute(
                input_seq,
                baselines=torch.zeros_like(input_seq),
                return_convergence_delta=True
            )
            
            attr_accum += attr.squeeze(0)  # Accumulate attributions across samples
            delta_list.append(delta.item())  # Record the convergence delta
            sampled += 1
        
        if sampled >= n_samples:
            break

    avg_attr = attr_accum / n_samples  # Compute average attribution per time step and feature
    avg_attr = avg_attr.detach().cpu().numpy()
    
    return n_samples, avg_attr, delta_list


# In[35]:


n_samples, avg_attr, delta_list = avg_ig_values(model, test_loader=test_loader)


# In[36]:


print(f"Average Convergence Delta: {np.mean(delta_list):.6f}")  # Print average convergence delta to diagnose IG


# In[37]:


plt.figure(figsize=(15, 10))
plt.imshow(avg_attr.T, aspect='auto', cmap='bwr_r', interpolation='nearest')
plt.colorbar(label='Average Attribution')
plt.xlabel("Time Step (0 = -1:55, 23 = -0:00)")
plt.ylabel("Features")
plt.title(f"Average Integrated Gradients Attribution ({n_samples} Samples)")
plt.xticks(range(24))
plt.yticks(range(6), sensor_names)
plt.grid(False)
plt.tight_layout()
plt.show()  # Heatmap of average attributions across time steps and features


# ### Inference

# In[32]:


df_test = pd.read_csv("./.data/test_aug.csv")  # Load test data for inference


# In[33]:


print(df_test)  # Inspect the test dataframe


# In[34]:


for col in df_test.columns:
    print(col)  # Print column names to verify features


# In[35]:


meta_data = ['id']
df_test_meta = df_test[meta_data]  # Extract meta information (ids) for submission
df_test.drop(columns=meta_data, inplace=True)  # Remove meta columns from feature set


# In[36]:


# Create dataset and data loader for inference (no targets)
infer_test_dataset = BrisT1DDataset(df_test, is_train=False)
infer_test_loader = DataLoader(infer_test_dataset, batch_size=256, shuffle=False)


# In[37]:


model.load_state_dict(torch.load('./.export/checkpoint-aug-tcn.pt'))  # Reload best model checkpoint


# In[38]:


model.eval()

all_preds = []
with torch.no_grad():
    for x_seq in tqdm(infer_test_loader, desc="Inference"):
        x_seq = x_seq.to(device)
        outputs = model(x_seq)
        all_preds.append(outputs.cpu().numpy())

all_preds = np.concatenate(all_preds)  # Concatenate predictions from all batches


# In[39]:


target_scaler = joblib.load('./.data/target_scaler_aug.pkl')
all_preds_original = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()  # Convert predictions back to original scale


# In[40]:


# Calibrate test predictions using the calibration model
calibrated_test_preds = calibration_model.predict(all_preds_original.reshape(-1, 1))
all_preds_original = calibrated_test_preds


# In[41]:


# Prepare the submission file
df_predictions = pd.DataFrame(all_preds_original, columns=["bg+1:00"])
df_results = pd.concat([df_test_meta.reset_index(drop=True), df_predictions], axis=1)
df_results.to_csv("./.export/test_submission_aug_tcn.csv", index=False)


# ### Submission

# In[42]:


get_ipython().system('kaggle competitions submit -c brist1d -f ./.export/test_submission_aug_tcn.csv -m "BrisT1D Submission: TCN (Calibrated)"')


# In[43]:


display(Image(filename='./.export/test_submission_aug_tcn.png'))  # Display the submission screenshot for confirmation

