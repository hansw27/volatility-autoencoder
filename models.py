import torch
import torch.nn as nn

class VolatilityAutoencoder(nn.Module):
    def __init__(self):
        super(VolatilityAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 3) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 50)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

class EarlyStopping:
    def __init__(self, patience=10, path='optimal_autoencoder.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            torch.save(model.state_dict(), self.path)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True