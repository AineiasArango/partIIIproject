import numpy as np
import torch
from torch import optim, nn, utils, Tensor
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from lightning import Trainer
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os

class ParticleDataset(Dataset):
    def __init__(self, data_list: List[Tuple[np.ndarray, np.ndarray]]):
        """
        data_list: List of tuples (X, y) where:
            X: array of shape (n_samples, n_particles, 5)
            y: array of shape (n_samples,)
        """
        self.data = []
        
        for X, y in data_list:
            X = torch.FloatTensor(X) if isinstance(X, np.ndarray) else X
            y = torch.FloatTensor(y) if isinstance(y, np.ndarray) else y
            self.data.extend(list(zip(X, y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CellMLP(nn.Module):
    def __init__(self, input_size=5, hidden_sizes=[64, 32]):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer (single value per cell)
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, num_cells, input_features)
        batch_size, num_cells, _ = x.shape
        
        # Reshape to process all cells through the same MLP
        x_reshaped = x.view(-1, x.shape[-1])
        
        # Process through MLP
        cell_outputs = self.network(x_reshaped)
        
        # Reshape back and sum over cells
        cell_outputs = cell_outputs.view(batch_size, num_cells)
        final_output = cell_outputs.sum(dim=1) / num_cells
        
        return final_output

class CellMLPLightning(L.LightningModule):
    def __init__(self, input_size=5, hidden_sizes=[64, 32], learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CellMLP(input_size, hidden_sizes)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

def load_and_prepare_data(data_paths: Dict[int, Tuple[str, str]]):
    """
    data_paths: Dictionary mapping particle count to tuple of (features_path, targets_path)
    Example:
    {
        32: ('path/to/32particle_features.npy', 'path/to/32particle_targets.npy'),
        64: ('path/to/64particle_features.npy', 'path/to/64particle_targets.npy')
    }
    """
    data_list = []
    
    for n_particles, (feature_path, target_path) in data_paths.items():
        # Load data
        X = np.load(feature_path)  # Expected shape: (n_samples, n_particles, 5)
        y = np.load(target_path)   # Expected shape: (n_samples,)
        
        print(f"Loaded {n_particles} particle data: {X.shape}")
        data_list.append((X, y))
    
    # Split into train and validation
    train_data = []
    val_data = []
    train_size = 0.8

    for X, y in data_list:
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split_idx = int(train_size * n_samples)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data.append((X[train_indices], y[train_indices]))
        val_data.append((X[val_indices], y[val_indices]))

    return train_data, val_data

def main():
    # Define your data paths
    data_paths = {
        32: ('path/to/32particle_features.npy', 'path/to/32particle_targets.npy'),
        64: ('path/to/64particle_features.npy', 'path/to/64particle_targets.npy')
    }
    
    # Load and prepare data
    train_data, val_data = load_and_prepare_data(data_paths)
    
    # Create datasets
    train_dataset = ParticleDataset(train_data)
    val_dataset = ParticleDataset(val_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = CellMLPLightning(
        input_size=5,
        hidden_sizes=[64, 32],
        learning_rate=1e-3
    )
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=100,
        accelerator='auto',  # Uses GPU if available
        devices=1,
        enable_progress_bar=True,
        callbacks=[
            L.callbacks.ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints',
                filename='best-model-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min'
            ),
            L.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    main()


