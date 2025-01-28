import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import random

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 50
batch_size = 100
learning_rate = 1e-3
num_epochs = 50  # Adjust as needed
label_counts = [100, 600, 1000, 3000]  # Number of labeled samples to use

# Transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    transform=transform
)

# Define the VAE model
# Encoder network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc31 = nn.Linear(600, latent_dim)  # For mean μ
        self.fc32 = nn.Linear(600, latent_dim)  # For log variance logσ^2

    def forward(self, x):
        h1 = F.softplus(self.fc1(x))
        h2 = F.softplus(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

# Decoder network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc4 = nn.Linear(latent_dim, 600)
        self.fc5 = nn.Linear(600, 600)
        self.fc6 = nn.Linear(600, 784)

    def forward(self, z):
        h4 = F.softplus(self.fc4(z))
        h5 = F.softplus(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

# VAE combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std  # Reparameterization trick

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# Loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Function to create labeled and unlabeled datasets
def get_labeled_unlabeled_loaders(label_count):
    # Ensure classes are balanced
    labels_per_class = label_count // 10
    labeled_idx = []
    unlabeled_idx = []
    class_counts = {k: 0 for k in range(10)}
    for idx, (data, target) in enumerate(train_dataset):
        if class_counts[target] < labels_per_class:
            labeled_idx.append(idx)
            class_counts[target] += 1
        else:
            unlabeled_idx.append(idx)
    # Labeled dataset
    labeled_subset = Subset(train_dataset, labeled_idx)
    labeled_loader = DataLoader(
        labeled_subset,
        batch_size=batch_size,
        shuffle=True
    )
    # Unlabeled dataset (excluding labeled data)
    unlabeled_subset = Subset(train_dataset, unlabeled_idx)
    unlabeled_loader = DataLoader(
        unlabeled_subset,
        batch_size=batch_size,
        shuffle=True
    )
    # Combined dataset for VAE training
    combined_subset = Subset(train_dataset, labeled_idx + unlabeled_idx)
    combined_loader = DataLoader(
        combined_subset,
        batch_size=batch_size,
        shuffle=True
    )
    return labeled_loader, unlabeled_loader, combined_loader

# Function to train the VAE
def train_vae(vae, optimizer, data_loader):
    vae.train()
    train_loss = 0
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        for data, _ in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        average_loss = epoch_loss / len(data_loader.dataset)
        train_loss = average_loss
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Loss: {average_loss:.4f}')
    return train_loss

# Function to extract latent representations
def extract_latent_representations(vae, data_loader):
    vae.eval()
    latent_vectors = []
    targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            mu, logvar = vae.encoder(data.view(-1, 784))
            z = vae.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu().numpy())
            targets.append(target.numpy())
    latent_vectors = np.concatenate(latent_vectors)
    targets = np.concatenate(targets)
    return latent_vectors, targets

# Main experiment function
def run_experiment():
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_latent_vectors = None
    test_targets = None
    for label_count in label_counts:
        print(f'\nRunning experiment with {label_count} labeled samples...')
        # Get data loaders
        labeled_loader, unlabeled_loader, combined_loader = get_labeled_unlabeled_loaders(label_count)
        # Initialize VAE and optimizer
        vae = VAE().to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
        # Train VAE on combined data (both labeled and unlabeled)
        print('Training VAE...')
        train_vae(vae, optimizer, combined_loader)
        # Extract latent representations for labeled data
        print('Extracting latent representations for labeled data...')
        labeled_latent_vectors, labeled_targets = extract_latent_representations(vae, labeled_loader)
        # Train SVM on labeled latent vectors
        print('Training SVM classifier...')
        clf = svm.SVC(kernel='rbf', gamma='scale')
        clf.fit(labeled_latent_vectors, labeled_targets)
        # Extract latent representations for test data (only once)
        if test_latent_vectors is None:
            print('Extracting latent representations for test data...')
            test_latent_vectors, test_targets = extract_latent_representations(vae, test_loader)
        # Predict and evaluate
        print('Evaluating classifier on test data...')
        predicted = clf.predict(test_latent_vectors)
        accuracy = accuracy_score(test_targets, predicted)
        print(f'Accuracy with {label_count} labels: {accuracy * 100:.2f}%')

# Run the experiment
if __name__ == '__main__':
    run_experiment()
