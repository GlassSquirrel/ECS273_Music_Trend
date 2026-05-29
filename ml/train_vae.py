"""
train_vae.py
============
Define and train the tri-modal Variational Autoencoder (VAE) on preprocessed
feature arrays, then extract and save latent vectors.

The VAE fuses three parallel encoders (acoustic, lyric, artist tags) into a
shared 32-dimensional latent space, which is later used for clustering.

Input (from ../data/processed/):
    acoustic.npy, lyric.npy, tags.npy
    has_lyrics.npy, has_tags.npy

Output:
    results/vae_weights.pt          -- trained model state dict
    results/latent_vectors.npy      -- (N, LATENT_DIM) μ vectors (deterministic)
    results/figures/training_history.png -- per-epoch loss curves

Usage (from the `ml/` directory):
    python train_vae.py
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
PROCESSED_DIR = "../data/processed"
RESULTS_DIR   = "results"          # output folder for weights and latents
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")  # output folder for all plots
LATENT_DIM    = 32
BATCH_SIZE    = 256
EPOCHS        = 100
LR            = 1e-3
BETA_MAX      = 0.5    # maximum KL weight (annealed from 0 over first 50 epochs)
RANDOM_SEED   = 42
ACOUSTIC_WEIGHT = 1.0
LYRIC_WEIGHT    = 1.0
TAGS_WEIGHT     = 0.5

logger = setup_logger("train_vae")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class MultimodalVAE(nn.Module):
    """
    Tri-modal Variational Autoencoder.

    Architecture
    ------------
    Three parallel encoders map their respective modalities to 64-dim vectors:
        Acoustic encoder : (acoustic_dim) → 128 → 64
        Lyric encoder    : (lyric_dim)    → 128 → 64
        Tags encoder     : (tags_dim)     → 128 → 64

    The three 64-dim outputs are concatenated → 192-dim fusion vector, then
    projected to the latent distribution parameters (μ, log σ²) of dimension
    LATENT_DIM.

    A single shared decoder reconstructs all three modalities jointly:
        LATENT_DIM → 256 → 512 → (acoustic_dim + lyric_dim + tags_dim)
    """

    def __init__(self, acoustic_dim: int, lyric_dim: int, tags_dim: int,
                 latent_dim: int):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.lyric_dim    = lyric_dim
        self.tags_dim     = tags_dim
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

        # Artist tags encoder
        self.tags_enc = nn.Sequential(
            nn.Linear(tags_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),       nn.BatchNorm1d(64),  nn.ReLU(),
        )

        # Projection heads for latent distribution parameters
        self.fc_mu     = nn.Linear(192, latent_dim)
        self.fc_logvar = nn.Linear(192, latent_dim)

        # Shared decoder reconstructs all three modalities
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512),        nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, acoustic_dim + lyric_dim + tags_dim),
        )

    def encode(self, x_acoustic, x_lyric, x_tags):
        """Encode inputs → (μ, log σ²)."""
        h_a = self.acoustic_enc(x_acoustic)
        h_l = self.lyric_enc(x_lyric)
        h_t = self.tags_enc(x_tags)
        h   = torch.cat([h_a, h_l, h_t], dim=1)  # (B, 192)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z via the reparameterisation trick (disabled at eval time)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # At inference, use the deterministic mean
        return mu

    def decode(self, z):
        """Decode latent vector z → reconstructed modality triplet."""
        out = self.decoder(z)
        split_a = self.acoustic_dim
        split_l = self.acoustic_dim + self.lyric_dim
        return out[:, :split_a], out[:, split_a:split_l], out[:, split_l:]

    def forward(self, x_acoustic, x_lyric, x_tags):
        mu, logvar = self.encode(x_acoustic, x_lyric, x_tags)
        z = self.reparameterize(mu, logvar)
        x_a_hat, x_l_hat, x_t_hat = self.decode(z)
        return x_a_hat, x_l_hat, x_t_hat, mu, logvar


# ──────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────

def vae_loss(x_a, x_l, x_t,
             x_a_hat, x_l_hat, x_t_hat,
             mu, logvar,
             lyric_mask, tags_mask,
             beta: float = 1.0,
             acoustic_weight: float = ACOUSTIC_WEIGHT,
             lyric_weight: float = LYRIC_WEIGHT,
             tags_weight: float = TAGS_WEIGHT):
    """
    Compute the β-VAE loss with modality-aware masking.

    Loss = acoustic_weight * MSE(acoustic)
         + lyric_weight    * MSE(lyric,  rows where lyrics available)
         + tags_weight     * BCE(tags,   rows where tags available)
         + beta            * KL(q(z|x) || p(z))

    Parameters
    ----------
    lyric_mask, tags_mask : bool tensors indicating data availability per sample
    beta                  : KL annealing weight
    acoustic_weight       : relative weight of acoustic reconstruction loss
    lyric_weight          : relative weight of lyric reconstruction loss
    tags_weight           : relative weight of tags reconstruction loss

    Returns
    -------
    total   : scalar loss tensor
    ra, rl, rt, kl : individual loss components (float) for logging
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

    if tags_mask.sum() > 0:
        recon_t = nn.functional.binary_cross_entropy_with_logits(
            x_t_hat[tags_mask], x_t[tags_mask], reduction="mean"
        )
    else:
        recon_t = torch.tensor(0.0, device=x_a.device)
    if torch.isnan(recon_t):
        recon_t = torch.tensor(0.0, device=x_a.device)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(kl):
        kl = torch.tensor(0.0, device=x_a.device)

    total = (acoustic_weight * recon_a
             + lyric_weight  * recon_l
             + tags_weight   * recon_t
             + beta          * kl)
    return total, recon_a.item(), recon_l.item(), recon_t.item(), kl.item()

# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_vae(model: MultimodalVAE, loader: DataLoader,
              epochs: int = EPOCHS, lr: float = LR,
              beta_max: float = BETA_MAX,
              acoustic_weight: float = ACOUSTIC_WEIGHT,
              lyric_weight: float = LYRIC_WEIGHT,
              tags_weight: float = TAGS_WEIGHT):
    """
    Train the VAE with KL annealing.

    Beta is linearly ramped from 0 → beta_max over the first 50% of epochs,
    allowing the model to prioritise reconstruction before regularising the
    latent space (prevents posterior collapse).

    Returns
    -------
    history : dict with per-epoch lists: 'loss', 'recon_a', 'recon_l',
              'recon_t', 'kl'
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Cosine annealing gently reduces LR toward 0 over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(DEVICE)

    history = {"loss": [], "recon_a": [], "recon_l": [], "recon_t": [], "kl": []}

    for epoch in range(1, epochs + 1):
        model.train()
        # KL annealing: β increases linearly in the first half of training
        beta = beta_max * min(1.0, epoch / (epochs * 0.5))

        epoch_loss = epoch_ra = epoch_rl = epoch_rt = epoch_kl = 0.0

        for x_a, x_l, x_t, has_lyr, has_trm in loader:
            x_a = x_a.to(DEVICE)
            x_l = x_l.to(DEVICE)
            x_t = x_t.to(DEVICE)
            has_lyr = has_lyr.bool().to(DEVICE)
            has_trm = has_trm.bool().to(DEVICE)

            optimizer.zero_grad()
            x_a_hat, x_l_hat, x_t_hat, mu, logvar = model(x_a, x_l, x_t)
            loss, ra, rl, rt, kl = vae_loss(
                x_a, x_l, x_t,
                x_a_hat, x_l_hat, x_t_hat,
                mu, logvar,
                has_lyr, has_trm,
                beta=beta,
                acoustic_weight=acoustic_weight,
                lyric_weight=lyric_weight,
                tags_weight=tags_weight,
            )
            loss.backward()
            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ra   += ra
            epoch_rl   += rl
            epoch_rt   += rt
            epoch_kl   += kl

        n = len(loader)
        history["loss"].append(epoch_loss / n)
        history["recon_a"].append(epoch_ra / n)
        history["recon_l"].append(epoch_rl / n)
        history["recon_t"].append(epoch_rt / n)
        history["kl"].append(epoch_kl / n)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                  f"loss={epoch_loss/n:.4f}  "
                  f"recon_a={epoch_ra/n:.4f}  "
                  f"recon_l={epoch_rl/n:.4f}  "
                  f"recon_t={epoch_rt/n:.4f}  "
                  f"kl={epoch_kl/n:.4f}  "
                  f"β={beta:.3f}")

    return history


# ──────────────────────────────────────────────
# Latent vector extraction
# ──────────────────────────────────────────────

@torch.no_grad()
def extract_latent(model: MultimodalVAE, loader: DataLoader) -> np.ndarray:
    """
    Run inference over the full dataset and return deterministic μ vectors.

    Using μ (rather than sampled z) gives stable, reproducible latent
    representations for downstream clustering.

    Returns
    -------
    latent_vectors : np.ndarray (N, latent_dim)
    """
    model.eval()
    mus = []
    for x_a, x_l, x_t, _, _ in loader:
        x_a = x_a.to(DEVICE)
        x_l = x_l.to(DEVICE)
        x_t = x_t.to(DEVICE)
        mu, _ = model.encode(x_a, x_l, x_t)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus).astype(np.float32)


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = "training_history.png"):
    """Plot and save the four per-epoch loss curves."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].plot(history["recon_a"])
    axes[0].set(title="Acoustic Reconstruction Loss", xlabel="Epoch")
    axes[1].plot(history["recon_l"], color="orange")
    axes[1].set(title="Lyric Reconstruction Loss", xlabel="Epoch")
    axes[2].plot(history["recon_t"], color="green")
    axes[2].set(title="Tags Reconstruction Loss", xlabel="Epoch")
    axes[3].plot(history["kl"], color="red")
    axes[3].set(title="KL Divergence", xlabel="Epoch")
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
    logger.info("Loading preprocessed feature arrays ...")
    acoustic  = np.load(os.path.join(PROCESSED_DIR, "acoustic.npy"))
    lyric     = np.load(os.path.join(PROCESSED_DIR, "lyric.npy"))
    tags      = np.load(os.path.join(PROCESSED_DIR, "tags.npy"))
    has_lyrics = np.load(os.path.join(PROCESSED_DIR, "has_lyrics.npy"))
    has_tags   = np.load(os.path.join(PROCESSED_DIR, "has_tags.npy"))

    logger.info(f"  acoustic  : {acoustic.shape}")
    logger.info(f"  lyric     : {lyric.shape}")
    logger.info(f"  tags      : {tags.shape}")

    # ── Build DataLoader ─────────────────────────────────────────────────────
    dataset = TensorDataset(
        torch.tensor(acoustic,   dtype=torch.float32),
        torch.tensor(lyric,      dtype=torch.float32),
        torch.tensor(tags,       dtype=torch.float32),
        torch.tensor(has_lyrics, dtype=torch.float32),
        torch.tensor(has_tags,   dtype=torch.float32),
    )
    # Fix the shuffle order across runs for full reproducibility
    # g = torch.Generator()
    # g.manual_seed(RANDOM_SEED)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
        # generator=g,
    )

    # ── Initialise model ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Initialising MultimodalVAE ...")
    acoustic_dim = acoustic.shape[1]
    lyric_dim    = lyric.shape[1]
    tags_dim     = tags.shape[1]
    model = MultimodalVAE(acoustic_dim, lyric_dim, tags_dim, LATENT_DIM)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters : {n_params:,}")
    logger.info(f"  Input dims           : acoustic={acoustic_dim}, "
          f"lyric={lyric_dim}, tags={tags_dim}")
    logger.info(f"  Latent dim           : {LATENT_DIM}")

    # ── Create results directory ─────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"Training for {EPOCHS} epochs ...")
    logger.info(f"Loss weights: acoustic={ACOUSTIC_WEIGHT}, lyric={LYRIC_WEIGHT}, tags={TAGS_WEIGHT}")
    history = train_vae(model, loader, epochs=EPOCHS, lr=LR, beta_max=BETA_MAX,
                    acoustic_weight=ACOUSTIC_WEIGHT,
                    lyric_weight=LYRIC_WEIGHT,
                    tags_weight=TAGS_WEIGHT)
    plot_training_history(history,
                          save_path=os.path.join(FIGURES_DIR, "training_history.png"))

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