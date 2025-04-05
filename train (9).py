# Código adaptado para correr en Google Colab o Jupyter Notebook con TensorBoard

from google.colab import drive
drive.mount('/content/drive')

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models.vae import BetaVAE

# Reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# Configuración general
folds_dir = '/content/drive/MyDrive/1st_paper'
num_folds = 5
# Hiperparámetros "óptimos" obtenidos (ejemplo)
epochs = 1200          # O puedes subirlo a 100 o 200 según tus recursos
batch_size = 32      # Cambiado a 16
latent_dim = 512     # Igual que sugerido
hidden_dim = 2048     # Igual que sugerido
beta = 2.3547508300520814  # Aproximado
dropout = 0
lr = 0.002330676388493474
weight_decay = 0.00000041488466205118
early_stop = 150
input_channels = 4

# Función de pérdida
def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kld
    return total_loss, recon_loss.item(), kld.item()

# Entrenamiento y evaluación
def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = 0
    for x, in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, _ = model(x)
        loss, _, _ = loss_function(recon, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, device, beta):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, in loader:
            x = x.to(device)
            recon, mu, logvar, _ = model(x)
            loss, _, _ = loss_function(recon, x, mu, logvar, beta)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# Loop principal con TensorBoard
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando dispositivo: {device}')

for fold_idx in range(1, num_folds + 1):
    fold_path = os.path.join(folds_dir, f"fold_{fold_idx}")
    train_data = torch.load(os.path.join(fold_path, 'train_data_normed.pt'))
    val_data = torch.load(os.path.join(fold_path, 'val_data_normed.pt'))
    test_data = torch.load(os.path.join(fold_path, 'test_data_normed.pt'))

    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size)

    model = BetaVAE(latent_dim, hidden_dim, beta, dropout, input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    writer = SummaryWriter(log_dir=os.path.join(fold_path, 'runs'))

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, beta)
        val_loss = eval_epoch(model, val_loader, device, beta)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        print(f'Fold {fold_idx}, Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(fold_path, f"best_beta_vae_model_fold_{fold_idx}.pth"))
            print('Mejor modelo guardado.')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop:
                print('Early stopping.')
                break

    writer.close()

    # Evaluación en test
    model.load_state_dict(torch.load(os.path.join(fold_path, f"best_beta_vae_model_fold_{fold_idx}.pth")))
    test_loss = eval_epoch(model, test_loader, device, beta)
    print(f'Fold {fold_idx}: test_loss={test_loss:.4f}')

# Comando para ejecutar TensorBoard
print("Para visualizar resultados en TensorBoard ejecuta:")
print("%load_ext tensorboard")
print(f"%tensorboard --logdir '{folds_dir}'")
