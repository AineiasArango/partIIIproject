import numpy as np
import torch
from torch import optim, nn, utils, Tensor
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from lightning import Trainer
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
import torchmetrics

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
        
        # Output layer (single binary output)
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())  # Added sigmoid for binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, num_cells, input_features)
        batch_size, num_cells, _ = x.shape
        
        # Reshape to process all cells through the same MLP
        x_reshaped = x.view(-1, x.shape[-1])
        
        # Process through MLP
        cell_outputs = self.network(x_reshaped)
        
        # Reshape back and average over cells
        cell_outputs = cell_outputs.view(batch_size, num_cells)
        final_output = cell_outputs.mean(dim=1)  # Changed from sum to mean
        
        return final_output

class CellMLPLightning(L.LightningModule):
    def __init__(self, input_size=5, hidden_sizes=[64, 64, 32], learning_rate=1e-4, plot_name='training_metrics.png'):
        super().__init__()
        self.save_hyperparameters()
        self.model = CellMLP(input_size, hidden_sizes)
        self.learning_rate = learning_rate
        self.plot_name = plot_name
        self.hidden_sizes = hidden_sizes  # Store hidden_sizes for plot title
        
        # Initialize metric tracking for binary classification
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_auroc = torchmetrics.AUROC(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        
        # Storage for plotting
        self.epoch_train_losses = []    # Store one loss per epoch
        self.epoch_val_losses = []      # Store one loss per epoch
        self.current_train_losses = []   # Temporary storage for current epoch
        self.current_val_losses = []     # Temporary storage for current epoch
        
        # Add test metrics
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Store loss for current epoch
        self.current_train_losses.append(loss.item())
        
        # Calculate metrics
        accuracy = self.train_accuracy(y_hat, y)
        auroc = self.train_auroc(y_hat, y)
        f1 = self.train_f1(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_auroc', auroc, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        
        return {'loss': loss, 'accuracy': accuracy, 'auroc': auroc, 'f1': f1}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Store loss for current epoch
        self.current_val_losses.append(loss.item())
        
        # Calculate metrics
        accuracy = self.val_accuracy(y_hat, y)
        auroc = self.val_auroc(y_hat, y)
        f1 = self.val_f1(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_auroc', auroc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        
        return {'loss': loss, 'accuracy': accuracy, 'auroc': auroc, 'f1': f1}
    
    def on_train_epoch_end(self):
        # Average and store training loss for the epoch
        if self.current_train_losses:
            avg_loss = sum(self.current_train_losses) / len(self.current_train_losses)
            self.epoch_train_losses.append(avg_loss)
            self.current_train_losses = []  # Reset for next epoch
        
        avg_accuracy = self.train_accuracy.compute()
        avg_auroc = self.train_auroc.compute()
        avg_f1 = self.train_f1.compute()
        
        self.train_metrics_history['accuracy'].append(avg_accuracy.item())
        self.train_metrics_history['auroc'].append(avg_auroc.item())
        self.train_metrics_history['f1'].append(avg_f1.item())
        
        # Reset metrics
        self.train_accuracy.reset()
        self.train_auroc.reset()
        self.train_f1.reset()
    
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
            
        # Average and store validation loss for the epoch
        if self.current_val_losses:
            avg_loss = sum(self.current_val_losses) / len(self.current_val_losses)
            self.epoch_val_losses.append(avg_loss)
            self.current_val_losses = []  # Reset for next epoch
        
        avg_accuracy = self.val_accuracy.compute()
        avg_auroc = self.val_auroc.compute()
        avg_f1 = self.val_f1.compute()
        
        self.val_metrics_history['accuracy'].append(avg_accuracy.item())
        self.val_metrics_history['auroc'].append(avg_auroc.item())
        self.val_metrics_history['f1'].append(avg_f1.item())
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_auroc.reset()
        self.val_f1.reset()
    
    def plot_metrics(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Create title with model architecture and learning rate
        architecture_str = f"Hidden Sizes: {self.hidden_sizes}, Learning Rate: {self.learning_rate}"
        fig.suptitle(architecture_str, fontsize=12, y=1.02)
        
        # Plot losses using epoch-averaged values
        ax1.plot(self.epoch_train_losses, label='Training Loss')
        ax1.plot(self.epoch_val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Binary Cross Entropy Loss vs. Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Accuracy
        ax2.plot(self.train_metrics_history['accuracy'], label='Train Accuracy')
        ax2.plot(self.val_metrics_history['accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs. Epoch')
        ax2.legend()
        ax2.grid(True)
        
        # Plot AUROC
        ax3.plot(self.train_metrics_history['auroc'], label='Train AUROC')
        ax3.plot(self.val_metrics_history['auroc'], label='Validation AUROC')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AUROC')
        ax3.set_title('AUROC vs. Epoch')
        ax3.legend()
        ax3.grid(True)
        
        # Plot F1 Score
        ax4.plot(self.train_metrics_history['f1'], label='Train F1')
        ax4.plot(self.val_metrics_history['f1'], label='Validation F1')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score vs. Epoch')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plot_name, bbox_inches='tight')
        plt.close()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        
        # Calculate metrics
        accuracy = self.test_accuracy(y_hat, y)
        auroc = self.test_auroc(y_hat, y)
        f1 = self.test_f1(y_hat, y)
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        self.log('test_auroc', auroc)
        self.log('test_f1', f1)
        
        return {'test_loss': loss, 'test_accuracy': accuracy, 'test_auroc': auroc, 'test_f1': f1}

def load_and_prepare_data(data_paths: Dict[int, Tuple[str, str]]):
    """
    data_paths: Dictionary mapping particle count to tuple of (features_path, targets_path)
    """
    train_data_dict = {}
    val_data_dict = {}
    test_data_dict = {}  # Added test dictionary
    train_size = 0.7  # Adjusted to account for test set
    val_size = 0.15
    test_size = 0.15
    
    for n_particles, (feature_path, target_path) in data_paths.items():
        # Load data
        X = np.load(feature_path)
        y = np.load(target_path)
        
        # Transform y values: 1 if negative, 0 otherwise
        y = (y < 0).astype(float)
        
        print(f"Loaded {n_particles} particle data: {X.shape}")
        
        # Split into train, validation, and test
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        train_end = int(train_size * n_samples)
        val_end = int((train_size + val_size) * n_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]  # Test indices
        
        train_data_dict[n_particles] = (X[train_indices], y[train_indices])
        val_data_dict[n_particles] = (X[val_indices], y[val_indices])
        test_data_dict[n_particles] = (X[test_indices], y[test_indices])  # Test data

    return train_data_dict, val_data_dict, test_data_dict

def main(hidden_sizes=[64, 32], learning_rate=1e-4, epochs=100, plot_name='training_metrics.png'):
    # Define your data paths
    data_paths = {
        32: ('/data/ERCblackholes4/aasnha2/for_aineias/plots/X_32.npy', '/data/ERCblackholes4/aasnha2/for_aineias/plots/y_32.npy'),
        #64: ('/data/ERCblackholes4/aasnha2/for_aineias/plots/X_64.npy', '/data/ERCblackholes4/aasnha2/for_aineias/plots/y_64.npy'),
        #128: ('/data/ERCblackholes4/aasnha2/for_aineias/plots/X_128.npy', '/data/ERCblackholes4/aasnha2/for_aineias/plots/y_128.npy'),
        #256: ('/data/ERCblackholes4/aasnha2/for_aineias/plots/X_256.npy', '/data/ERCblackholes4/aasnha2/for_aineias/plots/y_256.npy')
    }
    
    # Load and prepare data
    train_data_dict, val_data_dict, test_data_dict = load_and_prepare_data(data_paths)
    
    # Create separate datasets and dataloaders
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}  # Added test loaders
    batch_size = 64
    
    for n_particles in data_paths.keys():
        train_dataset = ParticleDataset([(train_data_dict[n_particles][0], train_data_dict[n_particles][1])])
        val_dataset = ParticleDataset([(val_data_dict[n_particles][0], val_data_dict[n_particles][1])])
        test_dataset = ParticleDataset([(test_data_dict[n_particles][0], test_data_dict[n_particles][1])])  # Test dataset
        
        train_loaders[n_particles] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loaders[n_particles] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loaders[n_particles] = DataLoader(  # Test dataloader
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    
    # Initialize model and trainer
    model = CellMLPLightning(
        input_size=5,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        plot_name=plot_name
    )
    
    trainer = Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=1,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints',
                filename='best-model-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min'
            )
        ]
    )
    
    # Train and test for each particle count
    for n_particles, train_loader in train_loaders.items():
        print(f"\nTraining on {n_particles} particle data")
        
        # Reset metrics history before training
        model.current_epoch_train_losses = []
        model.current_epoch_val_losses = []
        model.training_losses = []
        model.validation_losses = []
        model.train_metrics_history = {'accuracy': [], 'auroc': [], 'f1': []}
        model.val_metrics_history = {'accuracy': [], 'auroc': [], 'f1': []}
        
        # Train the model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loaders[n_particles]
        )
        
        # Test the model
        test_results = trainer.test(model, test_loaders[n_particles])
        print(f"\nTest Results for {n_particles} particles:")
        print(f"Test Loss: {test_results[0]['test_loss']:.4f}")
        print(f"Test Accuracy: {test_results[0]['test_accuracy']:.4f}")
        print(f"Test AUROC: {test_results[0]['test_auroc']:.4f}")
        print(f"Test F1: {test_results[0]['test_f1']:.4f}")
        
        # Plot metrics
        model.plot_metrics()

if __name__ == "__main__":
    main(
        hidden_sizes=[64, 64, 32],
        learning_rate=1e-3,
        epochs=100,
        plot_name='my_custom_plot1.png'  # Example of custom plot name
    )
    main(
        hidden_sizes=[64, 32],
        learning_rate=1e-3,
        epochs=100,
        plot_name='my_custom_plot2.png'  # Example of custom plot name
    )
    main(
        hidden_sizes=[64, 64,32],
        learning_rate=1e-4,
        epochs=1000,
        plot_name='my_custom_plot3.png'  # Example of custom plot name
    )
    main(
        hidden_sizes=[64, 32],
        learning_rate=1e-4,
        epochs=1000,
        plot_name='my_custom_plot4.png'  # Example of custom plot name
    )
