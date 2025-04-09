import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import wandb
import matplotlib.pyplot as plt
import io
import io
import matplotlib.pyplot as plt
from PIL import Image  # <-- Important!
import wandb

from models.vae import BetaVAE  # Ensure you have BetaVAE accessible here

# -------------------------------
# GLOBAL SEED for reproducibility
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------
# 1) BETA WARM-UP SCHEDULER
# -----------------------------------------------------
def linear_beta_schedule(epoch, warmup_epochs, max_beta):
    """
    Linearly increases beta from 0 to max_beta over warmup_epochs.
    After warmup, stays at max_beta.
    """
    if epoch < warmup_epochs:
        return max_beta * (epoch / warmup_epochs)
    return max_beta

# -----------------------------------------------------
# 2) LOSS FUNCTION
# -----------------------------------------------------
def loss_function(recon_x, x, mu, logvar, beta):
    """
    Beta-VAE loss with MSE reconstruction and a KL term scaled by beta.
    Returns (total_loss, recon_loss, kl_loss).
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # MSE over all elements
    # KL = 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
    # We typically do -0.5 * sum(...) / batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon_loss + beta * kld
    return total, recon_loss, kld

# -----------------------------------------------------
# 3) TRAIN / EVAL EPOCH
# -----------------------------------------------------
def train_epoch(model, loader, optimizer, device, current_beta):
    model.train()
    total_loss, total_recon, total_kld = 0.0, 0.0, 0.0

    for x, in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, _ = model(x)
        loss, rloss, kldloss = loss_function(recon, x, mu, logvar, current_beta)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss  += loss.item()  * bs
        total_recon += rloss.item() * bs
        total_kld   += kldloss.item() * bs

    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kld/n

def eval_epoch(model, loader, device, current_beta):
    model.eval()
    total_loss, total_recon, total_kld = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x, in loader:
            x = x.to(device)
            recon, mu, logvar, _ = model(x)
            loss, rloss, kldloss = loss_function(recon, x, mu, logvar, current_beta)

            bs = x.size(0)
            total_loss  += loss.item()  * bs
            total_recon += rloss.item() * bs
            total_kld   += kldloss.item() * bs

    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kld/n

# -----------------------------------------------------
# 4) LOG RECON IMAGES (4-channel data)
# -----------------------------------------------------


def log_reconstruction_examples(model, data_tensor, device, fold_idx, step, max_images=4):
    """
    - Takes up to 'max_images' examples from data_tensor.
    - Logs side-by-side original vs reconstructed images to W&B as PIL images.
    - Because we have 4 channels, we either:
       (A) log each channel as grayscale, or
       (B) pick just channel 0 to visualize, etc.

    We'll do a simple approach: log each channel 0 as a grayscale image
    for original & reconstruction.
    """

    model.eval()
    if data_tensor.size(0) < max_images:
        max_images = data_tensor.size(0)

    # Grab a slice
    x = data_tensor[:max_images].to(device)
    with torch.no_grad():
        recon, mu, logvar, _ = model(x)

    images_to_log = []
    for i in range(max_images):
        orig_ch0  = x[i, 0].detach().cpu().numpy()
        recon_ch0 = recon[i, 0].detach().cpu().numpy()

        # Create side-by-side figure
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(orig_ch0, cmap='gray')
        axs[0].set_title("Original (ch0)")
        axs[0].axis('off')

        axs[1].imshow(recon_ch0, cmap='gray')
        axs[1].set_title("Reconstructed (ch0)")
        axs[1].axis('off')

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)

        # Rewind and open as PIL Image
        buf.seek(0)
        pil_img = Image.open(buf)

        # Now W&B can accept it as a wandb.Image
        image_to_log = wandb.Image(pil_img, caption=f"Fold {fold_idx}, Sample {i}")
        images_to_log.append(image_to_log)

    wandb.log({f"fold_{fold_idx}/recon_samples": images_to_log}, step=step)



# -----------------------------------------------------
# 5) LOG LATENT DISTRIBUTIONS AS W&B TABLE
# -----------------------------------------------------
def log_latent_distributions(model, data_loader, device, fold_idx, step):
    """
    Goes through the data_loader, collects mu/logvar for each sample,
    logs them in a wandb.Table. This can help you do offline analysis
    or quick cluster checks. If latent_dim is large, be mindful of memory usage.
    """

    model.eval()
    latents = []
    with torch.no_grad():
        for x, in data_loader:
            x = x.to(device)
            _, mu, logvar, z = model(x)
            mu_np = mu.detach().cpu().numpy()
            logvar_np = logvar.detach().cpu().numpy()
            for i in range(mu_np.shape[0]):
                latents.append([ *mu_np[i], *logvar_np[i] ])

    # Suppose we label columns as mu_0..mu_{latent_dim-1}, logvar_0..logvar_{latent_dim-1}
    table_columns = []
    table_columns.extend([f"mu_{i}" for i in range(model.latent_dim)])
    table_columns.extend([f"logvar_{i}" for i in range(model.latent_dim)])

    table = wandb.Table(columns=table_columns)
    for row in latents:
        table.add_data(*row)

    wandb.log({f"fold_{fold_idx}/latent_distributions": table}, step=step)

# -----------------------------------------------------
# 6) SAVE LATENT SPACE (MU) TO CSV WITH LABELS
# -----------------------------------------------------
def save_latent_space(model, data_tensor, labels, device, out_csv_path):
    """
    For each sample in data_tensor, compute mu (latent mean) with the best model.
    Then save a CSV with shape (N, latent_dim+1) or (N, latent_dim+2) if
    you also want logvar.  We'll at least do:

        label, mu_0, mu_1, ..., mu_{latent_dim-1}

    so we can do PCA/UMAP/t-SNE externally.

    :param labels: a list (or tensor) of length N with e.g. subjectID or class label
    """
    model.eval()

    latents = []
    n = data_tensor.size(0)
    with torch.no_grad():
        for i in range(n):
            x = data_tensor[i].unsqueeze(0).to(device)  # shape (1, C, H, W)
            _, mu, logvar, _ = model(x)
            mu_np = mu.squeeze(0).cpu().numpy()  # shape (latent_dim,)
            label_i = labels[i]  # string or numeric
            latents.append((label_i, mu_np))

    # Save as CSV: label, mu_0, mu_1, ..., mu_{dim-1}
    # or if label is numeric, you can parse it accordingly
    with open(out_csv_path, "w") as f:
        # Write header
        latent_dim = mu_np.shape[0]
        header_cols = ["label"] + [f"mu_{i}" for i in range(latent_dim)]
        f.write(",".join(header_cols) + "\n")
        # Write each row
        for label_val, mu_vector in latents:
            row_str = f"{label_val}," + ",".join([f"{v:.6f}" for v in mu_vector])
            f.write(row_str + "\n")

    print(f"[INFO] Saved latent space to {out_csv_path}")


# -----------------------------------------------------
# 7) TRAIN FUNCTION WITH CROSS-VALIDATION + BETA WARMUP
# -----------------------------------------------------
def train_one_run(config=None):
    """
    Runs cross-validation with the given hyperparams (from wandb.config),
    logs metrics to Weights & Biases in a single run.
    Includes:
      - Beta warmup
      - Logging reconstruction images
      - Logging latent distributions
      - Potential to log MIG/DCI, etc.
      - Saving latent means + labels to CSV for PCA/UMAP/t-SNE
    """
    with wandb.init(config=config):
        cfg = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        folds_dir   = cfg.get("folds_dir", "/content/drive/MyDrive/1st_paper")
        num_folds   = cfg.get("num_folds", 5)
        early_stop  = cfg.get("early_stop", 120)
        warmup_ep   = cfg.get("warmup_epochs", 50)   # how many epochs to warm up beta

        fold_test_losses = []

        for fold_idx in range(1, num_folds+1):
            print(f"\n=== [Fold {fold_idx}] ===")
            fold_path = os.path.join(folds_dir, f"fold_{fold_idx}")

            # ----------------------------------
            # LOAD DATA & LABELS
            # ----------------------------------
            train_data = torch.load(
                os.path.join(fold_path, 'train_data_normed.pt'),
                weights_only=False
            )
            train_label = torch.load(
                os.path.join(fold_path, f"train_labels_fold_{fold_idx}.pt"),
                weights_only=False
            )
            val_data = torch.load(
                os.path.join(fold_path, 'val_data_normed.pt'),
                weights_only=False
            )
            val_label = torch.load(
                os.path.join(fold_path, f"val_labels_fold_{fold_idx}.pt"),
                weights_only=False
            )
            test_data = torch.load(
                os.path.join(fold_path, 'test_data_normed.pt'),
                weights_only=False
            )
            test_label = torch.load(
                os.path.join(fold_path, f"test_labels_fold_{fold_idx}.pt"),
                weights_only=False
            )


            train_loader = DataLoader(TensorDataset(train_data), batch_size=cfg.batch_size, shuffle=True)
            val_loader   = DataLoader(TensorDataset(val_data),   batch_size=cfg.batch_size)
            test_loader  = DataLoader(TensorDataset(test_data),  batch_size=cfg.batch_size)

            # ----------------------------------
            # MODEL
            # ----------------------------------
            model = BetaVAE(
                latent_dim   = cfg.latent_dim,
                hidden_dim   = cfg.hidden_dim,
                beta         = cfg.beta,  # Might or might not be used inside the model forward
                dropout_rate = cfg.dropout,
                input_channels = cfg.input_channels
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

            best_val_loss = float('inf')
            epochs_no_improve = 0

            # Pre-sample some data for logging images
            sample_train_data = train_data[:8].clone()  # just a small batch for recon logging

            # ----------------------------------
            # TRAINING LOOP
            # ----------------------------------
            for epoch in range(1, cfg.epochs+1):
                # 1) Compute current beta based on warmup
                current_beta = linear_beta_schedule(epoch, warmup_ep, cfg.beta)

                # 2) Train one epoch
                train_loss, train_recon, train_kld = train_epoch(model, train_loader, optimizer, device, current_beta)
                # 3) Validate
                val_loss, val_recon, val_kld       = eval_epoch(model,   val_loader,   device, current_beta)

                # 4) Log metrics
                wandb.log({
                    f"fold_{fold_idx}/train_loss": train_loss,
                    f"fold_{fold_idx}/val_loss":   val_loss,
                    f"fold_{fold_idx}/train_recon": train_recon,
                    f"fold_{fold_idx}/val_recon":   val_recon,
                    f"fold_{fold_idx}/train_kld": train_kld,
                    f"fold_{fold_idx}/val_kld":   val_kld,
                    f"fold_{fold_idx}/beta":      current_beta,
                    "epoch": epoch
                })

                print(f"[Fold {fold_idx} | Epoch {epoch}/{cfg.epochs} | beta={current_beta:.3f}] "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # 5) Step scheduler
                scheduler.step(val_loss)

                # 6) Check for best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    save_path = os.path.join(fold_path, f"best_beta_vae_model_fold_{fold_idx}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"   -> New best model saved: {save_path}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stop:
                        print("   -> Early stopping triggered.")
                        break

                # 7) Optionally log reconstruction images every, say, 20 epochs
                if (epoch % 20) == 0:
                    log_reconstruction_examples(
                        model=model,
                        data_tensor=sample_train_data,
                        device=device,
                        fold_idx=fold_idx,
                        step=epoch,
                        max_images=4
                    )

            # ----------------------------------
            # EVALUATION ON TEST SET
            # ----------------------------------
            model.load_state_dict(torch.load(os.path.join(fold_path, f"best_beta_vae_model_fold_{fold_idx}.pth")))
            # We can also do a final pass logging images from test_data
            log_reconstruction_examples(
                model=model,
                data_tensor=test_data,
                device=device,
                fold_idx=fold_idx,
                step=cfg.epochs + fold_idx*1000,  # unique step
                max_images=4
            )

            # Optionally log latent distributions from the test set
            log_latent_distributions(model, test_loader, device, fold_idx, step=cfg.epochs + fold_idx*1000)

            # Now measure final test loss (using final beta)
            final_beta = cfg.beta  # after warmup, we want to see final
            test_loss, test_recon, test_kld = eval_epoch(model, test_loader, device, final_beta)
            fold_test_losses.append(test_loss)

            wandb.log({
                f"fold_{fold_idx}/test_loss": test_loss,
                f"fold_{fold_idx}/test_recon": test_recon,
                f"fold_{fold_idx}/test_kld": test_kld
            })

            print(f"[Fold {fold_idx}] final test_loss={test_loss:.4f}")

            # ----------------------------------
            # SAVE LATENT SPACE FOR PCA/UMAP
            # ----------------------------------
            # We'll do it for train, val, and test sets if we want complete coverage
            # (You can skip if not needed).
            train_latent_csv = os.path.join(fold_path, f"train_latent_space_fold_{fold_idx}.csv")
            val_latent_csv   = os.path.join(fold_path, f"val_latent_space_fold_{fold_idx}.csv")
            test_latent_csv  = os.path.join(fold_path, f"test_latent_space_fold_{fold_idx}.csv")

            save_latent_space(model, train_data, train_label, device, train_latent_csv)
            save_latent_space(model, val_data,   val_label,   device, val_latent_csv)
            save_latent_space(model, test_data,  test_label,  device, test_latent_csv)

        # Summarize all folds
        avg_test_loss = np.mean(fold_test_losses)
        wandb.log({"avg_test_loss": avg_test_loss})
        print("\n===== Cross-validation complete =====")
        for i, tl in enumerate(fold_test_losses, start=1):
            print(f"Fold {i}: Test Loss = {tl:.4f}")
        print(f"Average Test Loss: {avg_test_loss:.4f}")


# -----------------------------------------------------
# 8) MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    # Example usage: run a single run with default_config
    default_config = dict(
        folds_dir      = "/content/drive/MyDrive/1st_paper",
        num_folds      = 5,
        epochs         = 300,
        batch_size     = 16,
        latent_dim     = 512,
        hidden_dim     = 8192,
        beta           = 0.3,      # final beta
        dropout        = 0,
        lr             = 0.0008,
        weight_decay   = 0.00085,
        input_channels = 4,
        early_stop     = 49,
        warmup_epochs  = 30        # warm up beta over first 30 epochs
    )
    train_one_run(config=default_config)

