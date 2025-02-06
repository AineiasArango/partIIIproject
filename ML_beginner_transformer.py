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
from torchmetrics import F1Score

#import train and test data
data = np.load('/data/ERCblackholes4/aasnha2/for_aineias/plots/res_synthetic_data_all.npz')
x_train = torch.tensor(data['x_all_train'], dtype=torch.float32)
y_train = torch.tensor(data['y_all_train'], dtype=torch.float32)
x_test = torch.tensor(data['x_all_test'], dtype=torch.float32)
y_test = torch.tensor(data['y_all_test'], dtype=torch.float32)

# Split test data into validation and test sets
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


class Data(Dataset):
    def __init__(self, train=True, val=False):
        if train:
            self.x = x_train
            self.y = y_train
        elif val:
            self.x = x_val
            self.y = y_val
        else:
            self.x = x_test
            self.y = y_test
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

#Balanced data loader to deal with class imbalances
class BalancedDataLoader:
    def __init__(self, dataset, batch_size, num_workers=0):
        # Get positive and negative indices
        self.positive_indices = torch.where(dataset.y == 1)[0]
        self.negative_indices = torch.where(dataset.y == 0)[0]
        
        # Store dataset reference and batch size
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Calculate samples per class per batch
        self.samples_per_class = batch_size // 2
        
        # Calculate number of batches
        self.n_batches = min(len(self.positive_indices), len(self.negative_indices)) // self.samples_per_class
        
    def __iter__(self):
        # Shuffle indices for each epoch
        pos_indices = self.positive_indices[torch.randperm(len(self.positive_indices))]
        neg_indices = self.negative_indices[torch.randperm(len(self.negative_indices))]
        
        for i in range(self.n_batches):
            # Get batch indices
            pos_idx = pos_indices[i * self.samples_per_class:(i + 1) * self.samples_per_class]
            neg_idx = neg_indices[i * self.samples_per_class:(i + 1) * self.samples_per_class]
            
            # Combine and shuffle batch indices
            batch_indices = torch.cat([pos_idx, neg_idx])
            batch_indices = batch_indices[torch.randperm(len(batch_indices))]
            
            # Get batch data
            batch_x = self.dataset.x[batch_indices]
            batch_y = self.dataset.y[batch_indices]
            
            yield batch_x, batch_y
    
    def __len__(self):
        return self.n_batches

#design the network
class ParticleTransformer(L.LightningModule):
    def __init__(self, n_particles=32, n_features=5, n_heads=4, dropout_rate=0.3):
        super(ParticleTransformer, self).__init__()
        self.n_particles = n_particles
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        
        # Embedding layer
        self.embedding = nn.Linear(n_features, 64)
        self.weight_decay = 1e-5
        
        # Add Layer Normalization
        self.layer_norm = nn.LayerNorm(64)
        
        # Transformer with more regularization
        self.transformer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True  # Apply normalization before attention
        )
        
        # Output layers with better regularization
        self.output = nn.Sequential(
            nn.Linear(64 * n_particles, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        # Add lists to store losses and metrics for plotting
        self.training_losses = []
        self.validation_losses = []
        self.training_f1 = []
        self.validation_f1 = []
        
        # Initialize F1 Score metric
        self.f1 = F1Score(task="binary")
    
    def forward(self, x):
        # Reshape input: (batch, 160) -> (batch, 32, 5)
        x = x.view(-1, self.n_particles, self.n_features)
        
        # Embed each particle
        x = self.embedding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Flatten and pass through output layers
        x = x.reshape(x.size(0), -1)
        return self.output(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        f1_score = self.f1(y_pred, y.int())
        self.training_step_outputs.append({"loss": loss, "f1": f1_score})
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        test_loss = F.binary_cross_entropy(y_pred, y)
        f1_score = self.f1(y_pred, y.int())
        self.log("test_loss", test_loss)
        self.log("test_f1", f1_score)
        return {"test_loss": test_loss, "test_f1": f1_score}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = F.binary_cross_entropy(y_pred, y)
        f1_score = self.f1(y_pred, y.int())
        self.validation_step_outputs.append({"loss": val_loss, "f1": f1_score})
        return val_loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_f1 = torch.stack([x["f1"] for x in self.training_step_outputs]).mean()
        print(f"Epoch {self.current_epoch} Training Loss: {avg_loss:.4f}, F1 Score: {avg_f1:.4f}")
        self.training_losses.append(avg_loss.item())
        self.training_f1.append(avg_f1.item())
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        avg_f1 = torch.stack([x["f1"] for x in self.validation_step_outputs]).mean()
        print(f"Epoch {self.current_epoch} Validation Loss: {avg_loss:.4f}, F1 Score: {avg_f1:.4f}")
        self.validation_losses.append(avg_loss.item())
        self.validation_f1.append(avg_f1.item())
        print("-" * 50)
        self.validation_step_outputs.clear()

    def predict(self, x):
        with torch.no_grad():
            probabilities = self(x)
            return (probabilities >= 0.5).float()

# Create data loaders
train_dataset = Data(train=True)
train_loader = BalancedDataLoader(dataset=train_dataset, batch_size=128)

# Keep validation and test loaders as regular DataLoaders
val_dataset = Data(train=False, val=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, num_workers=79, shuffle=False)

test_dataset = Data(train=False, val=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=79, shuffle=False)

num_epochs = 20

if __name__ == '__main__':
    neural_net = ParticleTransformer()
    trainer = Trainer(
        max_epochs=num_epochs, 
        fast_dev_run=False, 
        enable_progress_bar=True,
        logger=False  # Disable default logging
    )
    
    trainer.fit(neural_net, train_loader, val_loader)
    trainer.test(neural_net, test_loader)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(neural_net.training_losses)), neural_net.training_losses, label='Training Loss')
    plt.plot(range(len(neural_net.validation_losses)), neural_net.validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    #plt.savefig('/data/ERCblackholes4/aasnha2/for_aineias/plots/loss_plot4.png')
    plt.show()
    plt.close()
    
    # Make predictions and show accuracy
    with torch.no_grad():
        predictions = neural_net(x_test)
        predictions = (predictions >= 0.5).float()
    
    accuracy = (predictions == y_test).float().mean() * 100
    print(f"Final Test Accuracy: {accuracy:.2f}%")

    with torch.no_grad():
        predictions = neural_net(x_train)
        predictions = (predictions >= 0.5).float()
    
    accuracy = (predictions == y_train).float().mean() * 100
    print(f"Final Training Accuracy: {accuracy:.2f}%")
