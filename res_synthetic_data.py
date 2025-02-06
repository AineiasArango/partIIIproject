import numpy as np
import torch
from torch import optim, nn, utils, Tensor
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from lightning import Trainer
from torch.nn import TransformerEncoderLayer
import matplotlib.pyplot as plt

#hyperparameters
num_epochs = 10

def log_columns(data, column_indices):
    for i in column_indices:
        data[:,i] = np.log(data[:,i])
    return data

def unlog_columns(data, column_indices):
    for i in column_indices:
        data[:,i] = np.exp(data[:,i])
    return data

def z_score_normalize_all(data, mean, std):
    return (data - mean) / std

def z_score_denormalize_all(data, mean, std):
    return data * std + mean

def normalize_columns_all(data, means, stds):
    # Convert to numpy if tensor
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    return (data - means[None, :]) / stds[None, :]

def denormalize_columns_all(data, means, stds):
    return data * stds[None, :] + means[None, :]

def W_func(r, smoothing_length):
    import numpy as np
    x = r/smoothing_length
    result = np.zeros_like(x)
    
    # First condition: 0 <= r/h <= 1/2
    mask1 = (x <= 1/2) & (x >= 0)
    result[mask1] = (1 - 6*x[mask1]**2 + 6*x[mask1]**3) * 8/(np.pi*smoothing_length**3)
    
    # Second condition: 1/2 < r/h <= 1
    mask2 = (x > 1/2) & (x <= 1)
    result[mask2] = 2*(1-x[mask2])**3 * 8/(np.pi*smoothing_length**3)
    
    return result

def in_or_out_2(r, v_r, densities):
    import numpy as np

    smoothing_length = max(r)
    # Calculate kernel weights
    weights = W_func(r, smoothing_length)
    # Calculate mass flux
    mass_flux = np.sum(densities * weights * v_r) / np.sum(weights)

    if mass_flux < 0:
        return 1
    else:
        return 0

#import the data
data1 = np.load('/data/ERCblackholes4/aasnha2/for_aineias/plots/smooth_and_inout_results.npz')  
data2 = np.load('/data/ERCblackholes4/aasnha2/for_aineias/plots/smooth_and_inout_test_results_lowSNEff.npz')
smooth_results_highSNEff = data1['smooth_results']
inout_results_highSNEff = data1['inout_results']
smooth_results_lowSNEff = data2['smooth_results']
inout_results_lowSNEff = data2['inout_results']
# Combine the data from both simulations
smooth_results = np.concatenate((smooth_results_highSNEff, smooth_results_lowSNEff))
inout_results = np.concatenate((inout_results_highSNEff, inout_results_lowSNEff))

print(sum(inout_results==1))
print(sum(inout_results==0))
print(smooth_results.shape)

smooth_results = torch.tensor(smooth_results, dtype=torch.float32)
inout_results = torch.tensor(inout_results, dtype=torch.float32)
x_train, x_test, y_train, y_test = train_test_split(smooth_results, inout_results, test_size=0.3, random_state=42)
# Convert PyTorch tensors back to numpy arrays
x_train = x_train.numpy()
y_train = y_train.numpy()

# This function processes the data and adds synthetic data to it
def process_data_and_synth(x, y, num_copies=10, SD=0.01):
    # Convert input to numpy if it's a tensor
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
        
    # Process data to get radial coordinates
    radial_smooth_results = []
    #For each snapshot, calculate the radial distance, radial velocity. This leaves you with r, v_r, rho, T, m for each particle.
    for snapshot in x:
        positions = snapshot[:, :3]
        # Use numpy operations consistently
        r_norm = np.linalg.norm(positions, axis=1, keepdims=True)
        r_hat = positions / r_norm
        radial_velocities = np.sum(snapshot[:, 3:6] * r_hat, axis=1)
        radial_distances = np.sqrt(np.sum(positions**2, axis=1))
        radial_smooth_results.append(np.column_stack((
            radial_distances.reshape(-1,1), 
            radial_velocities.reshape(-1,1), 
            snapshot[:, 6:]
        )))
    
    # Log and normalize data
    logged_data = np.array([log_columns(snapshot, [0,2,3,4]) for snapshot in radial_smooth_results])
    means = np.mean(logged_data.reshape(-1, logged_data.shape[-1]), axis=0)
    stds = np.std(logged_data.reshape(-1, logged_data.shape[-1]), axis=0)
    normalized_data = normalize_columns_all(logged_data, means, stds)

    # Convert to torch tensors for synthetic data generation
    x_real = torch.tensor(normalized_data, dtype=torch.float32)
    y_real = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # Generate noisy copies
    noise = torch.randn(num_copies, *x_real.shape) * SD
    x_synth = x_real.unsqueeze(0) + noise
    x_synth = x_synth.reshape(-1, *x_real.shape[1:])
    y_synth = y_real.repeat(num_copies, 1)

    return torch.cat([x_real, x_synth]), torch.cat([y_real, y_synth])

x_all_train, y_all_train = process_data_and_synth(x_train, y_train, num_copies=20, SD=0.05)
x_all_test, y_all_test = process_data_and_synth(x_test, y_test, num_copies=20, SD=0.05)

print(x_all_train.shape)
print(y_all_train.shape)
print(x_all_test.shape)
print(y_all_test.shape)

#save the data
np.savez('/data/ERCblackholes4/aasnha2/for_aineias/plots/res_synthetic_data_all.npz', x_all_train=x_all_train, y_all_train=y_all_train, x_all_test=x_all_test, y_all_test=y_all_test)
