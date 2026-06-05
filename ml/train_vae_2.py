"""
train_vae_2modal.py
===================
Bi-modal VAE baseline: fuses acoustic + lyric features only (no artist tags).

This script is structurally identical to train_vae.py but removes the tags
encoder and tags reconstruction loss. It is used as an ablation baseline to
quantify the contribution of the artist tag modality (RQ1).

Input (from ../data/processed/):
    acoustic.npy, lyric.npy
    has_lyrics.npy

Output:
    results_2modal/vae_weights.pt
    results_2modal/latent_vectors.npy
    results_2modal/figures/training_history.png

Usage (from the `ml/` directory):
    python train_vae_2modal.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from logger import setup_logger

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROCESSED_DIR   = "../data/processed"
RESULTS_DIR     = "results_2modal"       # separate output dir to avoid overwriting tri-modal
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")
LATENT_DIM      = 32                     # same as tri-modal for fair comparison
BATCH_SIZE      = 256
EPOCHS          = 100
LR              = 1e-3
BETA_MAX        = 0.5
RANDOM_SEED     = 42
ACOUSTIC_WEIGHT = 1.2
LYRIC_WEIGHT    = 1.5
# No TAGS_WEIGHT — tags modality is excluded in this ablation

logger = setup_logger("train_vae_2modal")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class BimodalVAE(nn.Module):
    """
    Bi-modal Variational Autoencoder (acoustic + lyric only).

    Architecture
    ------------
    Two parallel encoders map their respective modalities to 64-dim vectors:
        Acoustic encoder : (acoustic_dim) → 128 → 64
        Lyric encoder    : (lyric_dim)    → 128 → 64

    The two 64-dim outputs are concatenated → 128-dim fusion vector, then
    projected to the latent distribution parameters (μ, log σ²) of dimension
    LATENT_DIM.

    A shared decoder reconstructs both modalities jointly:
        LATENT_DIM → 256 → 512 → (acoustic_dim + lyric_dim)

    Note: fusion dim is 128 (vs 192 in tri-modal) because the tags encoder
    is absent. All other hyperparameters are kept identical for a fair
    ablation comparison.
    """

    def __init__(self, acoustic_dim: int, lyric_dim: int, latent_dim: int):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.lyric_dim    = lyric_dim
        self.latent_dim   = latent_dim

        # Acoustic encoder
        self.acoustic_enc = nn.Sequential(
            nn.Linear(acoustic_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),           nn.BatchNorm1d(64),  nn.ReLU(),
        )

        # Lyric encoder
        self.lyric_enc = nn.Sequential(
            nn.Linear(lyric_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
        )

        # Projection heads: fusion dim = 128 (64 acoustic + 64 lyric)
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Shared decoder reconstructs acoustic + lyric only
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512),        nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, acoustic_dim + lyric_dim),
        )

    def encode(self, x_acoustic, x_lyric):
        """Encode inputs → (μ, log σ²)."""
        h_a = self.acoustic_enc(x_acoustic)
        h_l = self.lyric_enc(x_lyric)
        h   = torch.cat([h_a, h_l], dim=1)   # (B, 128)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z via the reparameterisation trick (disabled at eval time)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        """Decode latent vector z → reconstructed (acoustic, lyric)."""
        out = self.decoder(z)
        return out[:, :self.acoustic_dim], out[:, self.acoustic_dim:]

    def forward(self, x_acoustic, x_lyric):
        mu, logvar  = self.encode(x_acoustic, x_lyric)
        z           = self.reparameterize(mu, logvar)
        x_a_hat, x_l_hat = self.decode(z)
        return x_a_hat, x_l_hat, mu, logvar


# ──────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────

def vae_loss(x_a, x_l,
             x_a_hat, x_l_hat,
             mu, logvar,
             lyric_mask,
             beta: float = 1.0,
             acoustic_weight: float = ACOUSTIC_WEIGHT,
             lyric_weight: float = LYRIC_WEIGHT):
    """
    Bi-modal β-VAE loss with lyric availability masking.

    Loss = acoustic_weight * MSE(acoustic)
         + lyric_weight    * MSE(lyric, rows where lyrics available)
         + beta            * KL(q(z|x) || p(z))
    """
    recon_a = nn.functional.mse_loss(x_a_hat, x_a, reduction="mean")

    if lyric_mask.sum() > 0:
        recon_l = nn.functional.mse_loss(
            x_l_hat[lyric_mask], x_l[lyric_mask], reduction="mean"
        )
    else:
        recon_l = torch.tensor(0.0, device=x_a.device)
    if torch.isnan(recon_l):
        recon_l = torch.tensor(0.0, device=x_a.device)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(kl):
        kl = torch.tensor(0.0, device=x_a.device)

    total = (acoustic_weight * recon_a
             + lyric_weight  * recon_l
             + beta          * kl)
    return total, recon_a.item(), recon_l.item(), kl.item()


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_vae(model: BimodalVAE, loader: DataLoader,
              epochs: int = EPOCHS, lr: float = LR,
              beta_max: float = BETA_MAX,
              acoustic_weight: float = ACOUSTIC_WEIGHT,
              lyric_weight: float = LYRIC_WEIGHT):
    """
    Train the bi-modal VAE with KL annealing.
    Beta is linearly ramped from 0 → beta_max over the first 50% of epochs.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(DEVICE)

    history = {"loss": [], "recon_a": [], "recon_l": [], "kl": []}

    for epoch in range(1, epochs + 1):
        model.train()
        beta = beta_max * min(1.0, epoch / (epochs * 0.5))

        epoch_loss = epoch_ra = epoch_rl = epoch_kl = 0.0

        for x_a, x_l, has_lyr in loader:
            x_a     = x_a.to(DEVICE)
            x_l     = x_l.to(DEVICE)
            has_lyr = has_lyr.bool().to(DEVICE)

            optimizer.zero_grad()
            x_a_hat, x_l_hat, mu, logvar = model(x_a, x_l)
            loss, ra, rl, kl = vae_loss(
                x_a, x_l, x_a_hat, x_l_hat,
                mu, logvar, has_lyr,
                beta=beta,
                acoustic_weight=acoustic_weight,
                lyric_weight=lyric_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ra   += ra
            epoch_rl   += rl
            epoch_kl   += kl

        n = len(loader)
        history["loss"].append(epoch_loss / n)
        history["recon_a"].append(epoch_ra / n)
        history["recon_l"].append(epoch_rl / n)
        history["kl"].append(epoch_kl / n)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                        f"loss={epoch_loss/n:.4f}  "
                        f"recon_a={epoch_ra/n:.4f}  "
                        f"recon_l={epoch_rl/n:.4f}  "
                        f"kl={epoch_kl/n:.4f}  "
                        f"β={beta:.3f}")

    return history


# ──────────────────────────────────────────────
# Latent vector extraction
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_latent(model: BimodalVAE, loader: DataLoader) -> np.ndarray:
    """
    Run inference and return deterministic μ vectors.
    Using μ (not sampled z) for stable, reproducible clustering.
    """
    model.eval()
    mus = []
    for x_a, x_l, _ in loader:
        x_a = x_a.to(DEVICE)
        x_l = x_l.to(DEVICE)
        mu, _ = model.encode(x_a, x_l)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus).astype(np.float32)


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str):
    """Plot and save per-epoch loss curves (3 panels: acoustic, lyric, KL)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(history["recon_a"])
    axes[0].set(title="Acoustic Reconstruction Loss", xlabel="Epoch")
    axes[1].plot(history["recon_l"], color="orange")
    axes[1].set(title="Lyric Reconstruction Loss", xlabel="Epoch")
    axes[2].plot(history["kl"], color="red")
    axes[2].set(title="KL Divergence", xlabel="Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # ── Load preprocessed arrays ────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("Loading preprocessed feature arrays (acoustic + lyric only) ...")
    acoustic   = np.load(os.path.join(PROCESSED_DIR, "acoustic.npy"))
    lyric      = np.load(os.path.join(PROCESSED_DIR, "lyric.npy"))
    has_lyrics = np.load(os.path.join(PROCESSED_DIR, "has_lyrics.npy"))

    logger.info(f"  acoustic  : {acoustic.shape}")
    logger.info(f"  lyric     : {lyric.shape}")
    logger.info("  (artist tags excluded — bi-modal ablation)")

    # ── Build DataLoader ─────────────────────────────────────────────────────
    dataset = TensorDataset(
        torch.tensor(acoustic,   dtype=torch.float32),
        torch.tensor(lyric,      dtype=torch.float32),
        torch.tensor(has_lyrics, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ── Initialise model ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Initialising BimodalVAE ...")
    acoustic_dim = acoustic.shape[1]
    lyric_dim    = lyric.shape[1]
    model = BimodalVAE(acoustic_dim, lyric_dim, LATENT_DIM)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters : {n_params:,}")
    logger.info(f"  Input dims           : acoustic={acoustic_dim}, lyric={lyric_dim}")
    logger.info(f"  Latent dim           : {LATENT_DIM}")

    # ── Create output directories ────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"Training for {EPOCHS} epochs ...")
    logger.info(f"Loss weights: acoustic={ACOUSTIC_WEIGHT}, lyric={LYRIC_WEIGHT}")
    history = train_vae(model, loader, epochs=EPOCHS, lr=LR, beta_max=BETA_MAX,
                        acoustic_weight=ACOUSTIC_WEIGHT,
                        lyric_weight=LYRIC_WEIGHT)
    plot_training_history(
        history,
        save_path=os.path.join(FIGURES_DIR, "training_history.png")
    )

    weights_path = os.path.join(RESULTS_DIR, "vae_weights.pt")
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Saved: {weights_path}")

    # ── Extract latent vectors ───────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Extracting latent vectors (μ) ...")
    full_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    latent_vectors = extract_latent(model, full_loader)
    logger.info(f"  Latent vectors shape : {latent_vectors.shape}")

    latent_path = os.path.join(RESULTS_DIR, "latent_vectors.npy")
    np.save(latent_path, latent_vectors)
    logger.info(f"Saved: {latent_path}")


if __name__ == "__main__":
    main()