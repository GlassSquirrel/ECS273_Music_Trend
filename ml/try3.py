"""
Multi-modal Music VAE Pipeline
================================
数据来源: MSD subset (10,000 songs) + musiXmatch lyrics merge
流程: 预处理 → TF-IDF歌词特征 → 多模态VAE → 聚类 → 时间趋势分析

依赖安装:
    pip install pandas numpy scikit-learn torch umap-learn matplotlib seaborn
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import umap
from sklearn.preprocessing import MaxAbsScaler
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────
DATA_PATH    = "../data/msd_mxm_merged.csv"
LATENT_DIM   = 32       # VAE latent space 维度
BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
N_CLUSTERS   = 10        # K-Means 聚类数（可调）
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. 数据读取 & 预处理
# ─────────────────────────────────────────────

def load_and_preprocess(path: str):
    """
    返回:
        acoustic_scaled  : np.ndarray (N, 29)  标准化后的音频特征
        lyric_tfidf      : np.ndarray (N, 100) TF-IDF 歌词特征（无歌词=全零）
        df_clean         : pd.DataFrame        原始行（过滤后）
    """
    df = pd.read_csv(path)
    print(f"原始数据: {df.shape}")

    # # ── 1.1 过滤年份缺失 ──────────────────────────────
    # df = df[df["year"] > 0].copy()
    # print(f"过滤 year=0 后: {df.shape}")

    # ── 1.2 音频特征选择 ──────────────────────────────
    # 注: danceability / energy 在 MSD subset 中全为 0，不使用
    ACOUSTIC_COLS = (
        ["loudness", "tempo", "mode", "key", "duration",
         "key_confidence", "mode_confidence", "time_signature",
         "time_signature_confidence", "bars_count", "beats_count",
         "sections_count", "segments_count",
         "avg_segment_loudness_max", "avg_segment_loudness_start"]
        + [f"avg_timbre_{i}" for i in range(12)]
        # avg_pitch 因为 pitch class 是循环的，建议编码后再用，这里暂时加入
        + [f"avg_pitch_{i}" for i in range(12)]
    )

    # 删除 acoustic 有空值的行（实际上 MSD subset 无缺失）
    df = df.dropna(subset=ACOUSTIC_COLS).reset_index(drop=True)

    # ── 1.3 标准化音频特征 ────────────────────────────
    scaler = StandardScaler()
    acoustic_scaled = scaler.fit_transform(df[ACOUSTIC_COLS].values.astype(np.float32))

    # ── 1.4 歌词 BoW 特征 ────────────────────────
    bow_cols = [c for c in df.columns if c.startswith("bow_")]
    lyric_tfidf = df[bow_cols].values.astype(np.float32)
    has_lyrics_mask = df["lyrics_available"] == 1
    tfidf = None

    # NaN检查（对原始BoW）
    if np.isnan(lyric_tfidf).any() or np.isinf(lyric_tfidf).any():
        print("Warning: NaN/Inf found in lyric features, replacing with 0")
        lyric_tfidf = np.nan_to_num(lyric_tfidf, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Acoustic features: {acoustic_scaled.shape}")
    print(f"Lyric BoW features (raw): {lyric_tfidf.shape}")
    print(f"Songs with lyrics: {has_lyrics_mask.sum()} / {len(df)}")

    # SVD压缩：1000维稀疏 → 50维dense
    svd = TruncatedSVD(n_components=50, random_state=RANDOM_SEED)
    lyric_dense = np.zeros((len(df), 50), dtype=np.float32)
    has_lyr = lyric_tfidf.sum(axis=1) > 0
    lyric_dense[has_lyr] = svd.fit_transform(lyric_tfidf[has_lyr])

    # ── 加这段：对SVD输出标准化，让scale和acoustic一致 ──
    lyric_scaler = StandardScaler()
    # 只用有歌词的行fit，避免全零行影响均值方差
    lyric_scaler.fit(lyric_dense[has_lyr])
    lyric_dense[has_lyr] = lyric_scaler.transform(lyric_dense[has_lyr])
    # 无歌词的行保持全零（标准化后有歌词行均值≈0，全零行不会太突兀）

    lyric_tfidf = lyric_dense

    print(f"Lyric dense features (after SVD): {lyric_tfidf.shape}")
    print(f"Lyric non-zero ratio after SVD: {(lyric_tfidf != 0).mean():.4f}")

    # ── 1.5 Artist Terms Multi-hot 编码 ──────────────
    def parse_terms(row):
        if isinstance(row["artist_terms"], str) and row["artist_terms"].strip():
            return [t.strip() for t in row["artist_terms"].split(";") if t.strip()]
        return []

    df["terms_list"] = df.apply(parse_terms, axis=1)

    # 统计所有标签频次，选最常见的TOP_TERMS个
    from collections import Counter
    all_terms = [t for terms in df["terms_list"] for t in terms]
    term_counts = Counter(all_terms)
    TOP_TERMS = 100  # 保留最常见的100个标签
    top_terms = [t for t, _ in term_counts.most_common(TOP_TERMS)]
    term2idx = {t: i for i, t in enumerate(top_terms)}

    print(f"Top 20 artist terms: {top_terms[:20]}")

    # 构建 multi-hot 矩阵
    terms_multihot = np.zeros((len(df), TOP_TERMS), dtype=np.float32)
    for i, terms in enumerate(df["terms_list"]):
        for t in terms:
            if t in term2idx:
                terms_multihot[i, term2idx[t]] = 1.0

    has_terms_mask = terms_multihot.sum(axis=1) > 0
    print(f"Artist terms features: {terms_multihot.shape}")
    print(f"Songs with terms: {has_terms_mask.sum()} / {len(df)}")
    print(f"Terms non-zero ratio: {(terms_multihot != 0).mean():.4f}")

    return acoustic_scaled, lyric_tfidf, terms_multihot, df, ACOUSTIC_COLS, scaler, tfidf


# ─────────────────────────────────────────────
# 2. 多模态 VAE 模型
# ─────────────────────────────────────────────

class MultimodalVAE(nn.Module):
    """
    两路编码器（音频 + 歌词）→ 共享 latent space → 解码器

    架构:
        Acoustic encoder:  29  → 128 → 64
        Lyric encoder:    100  → 128 → 64
        Fusion:          128  → μ (latent_dim), σ (latent_dim)
        Decoder:    latent_dim → 128 → 256 → 29+100
    """

    def __init__(self, acoustic_dim: int, lyric_dim: int, terms_dim: int, latent_dim: int):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.lyric_dim    = lyric_dim
        self.terms_dim    = terms_dim  # 加这行
        self.latent_dim   = latent_dim

        # ── Acoustic Encoder ──
        self.acoustic_enc = nn.Sequential(
            nn.Linear(acoustic_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),           nn.BatchNorm1d(64),  nn.ReLU(),
        )

        # ── Lyric Encoder ──
        self.lyric_enc = nn.Sequential(
            nn.Linear(lyric_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
        )

        # ── 新增 Terms Encoder ──
        self.terms_enc = nn.Sequential(
            nn.Linear(terms_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),  # 用参数terms_dim
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
        )

        # ── Fusion → μ, log_σ² ──
        self.fc_mu     = nn.Linear(192, latent_dim)
        self.fc_logvar = nn.Linear(192, latent_dim)

        # Decoder: 重建三路输出
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512),        nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, acoustic_dim + lyric_dim + terms_dim),  # 用参数
        )

    def encode(self, x_acoustic, x_lyric, x_terms):
        h_a = self.acoustic_enc(x_acoustic)
        h_l = self.lyric_enc(x_lyric)
        h_t = self.terms_enc(x_terms)
        h   = torch.cat([h_a, h_l, h_t], dim=1)  # (B, 192)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        out = self.decoder(z)
        return (out[:, :self.acoustic_dim],
                out[:, self.acoustic_dim:self.acoustic_dim + self.lyric_dim],
                out[:, self.acoustic_dim + self.lyric_dim:])

    def forward(self, x_acoustic, x_lyric, x_terms):
        mu, logvar = self.encode(x_acoustic, x_lyric, x_terms)
        z = self.reparameterize(mu, logvar)
        x_a_hat, x_l_hat, x_t_hat = self.decode(z)
        return x_a_hat, x_l_hat, x_t_hat, mu, logvar


# ─────────────────────────────────────────────
# 3. 损失函数
# ─────────────────────────────────────────────

def vae_loss(x_a, x_l, x_t, x_a_hat, x_l_hat, x_t_hat,
             mu, logvar, lyric_mask, terms_mask,
             beta=1.0, lyric_weight=0.1, terms_weight=0.3):
    """
    Total Loss = Reconstruction(acoustic) 
               + lyric_weight * Reconstruction(lyric, 仅有歌词的样本)
               + beta * KL散度

    lyric_mask: bool tensor，True 表示该样本有歌词
    beta:       KL 权重（beta-VAE 风格，初期可设小一些让 recon 先收敛）
    lyric_weight: 歌词重建损失权重（调小可降低无歌词样本的干扰）
    """
    recon_a = nn.functional.mse_loss(x_a_hat, x_a, reduction="mean")

    recon_l = (nn.functional.mse_loss(x_l_hat[lyric_mask], x_l[lyric_mask], reduction="mean")
               if lyric_mask.sum() > 0 else torch.tensor(0.0, device=x_a.device))
    if torch.isnan(recon_l): recon_l = torch.tensor(0.0, device=x_a.device)

    recon_t = (nn.functional.binary_cross_entropy_with_logits(
                   x_t_hat[terms_mask], x_t[terms_mask], reduction="mean")
               if terms_mask.sum() > 0 else torch.tensor(0.0, device=x_a.device))
    if torch.isnan(recon_t): recon_t = torch.tensor(0.0, device=x_a.device)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if torch.isnan(kl): kl = torch.tensor(0.0, device=x_a.device)

    total = recon_a + lyric_weight * recon_l + terms_weight * recon_t + beta * kl
    return total, recon_a.item(), recon_l.item(), recon_t.item(), kl.item()


# ─────────────────────────────────────────────
# 4. 训练循环
# ─────────────────────────────────────────────

def train_vae(model, loader, epochs=EPOCHS, lr=LR, beta_max=0.5):
    """
    KL 退火 (annealing): 前 50% epoch β 从 0 线性升至 beta_max
    这样模型先学好重建，再逐渐正则化 latent space。
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(DEVICE)

    history = {"loss": [], "recon_a": [], "recon_l": [], "recon_t": [], "kl": []}

    for epoch in range(1, epochs + 1):
        model.train()
        # KL annealing: β 在前半程线性增长
        beta = beta_max * min(1.0, (epoch / (epochs * 0.5)))

        epoch_loss = epoch_ra = epoch_rl = epoch_rt = epoch_kl = 0.0

        for x_a, x_l, x_t, has_lyr, has_trm in loader:
            x_a, x_l, x_t = x_a.to(DEVICE), x_l.to(DEVICE), x_t.to(DEVICE)
            has_lyr = has_lyr.bool().to(DEVICE)
            has_trm = has_trm.bool().to(DEVICE)

            optimizer.zero_grad()
            x_a_hat, x_l_hat, x_t_hat, mu, logvar = model(x_a, x_l, x_t)
            loss, ra, rl, rt, kl = vae_loss(x_a, x_l, x_t,
                                            x_a_hat, x_l_hat, x_t_hat,
                                            mu, logvar,
                                            has_lyr.bool(), has_trm.bool(),
                                            beta=beta, lyric_weight=1.0, terms_weight=0.5)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ra   += ra
            epoch_rl   += rl
            epoch_kl   += kl
            epoch_rt += rt

        n = len(loader)
        history["loss"].append(epoch_loss / n)
        history["recon_a"].append(epoch_ra / n)
        history["recon_l"].append(epoch_rl / n)
        history["recon_t"].append(epoch_rt / n)
        history["kl"].append(epoch_kl / n)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | loss={epoch_loss/n:.4f} "
                  f"recon_a={epoch_ra/n:.4f} recon_l={epoch_rl/n:.4f} "
                  f"recon_t={epoch_rt/n:.4f} kl={epoch_kl/n:.4f} β={beta:.3f}")

    return history


# ─────────────────────────────────────────────
# 5. 提取 Latent 向量
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_latent(model, loader):
    """推理模式下返回 μ（确定性 latent 向量，不加噪声）"""
    model.eval()
    mus = []
    for x_a, x_l, x_t, _, _ in loader:
        x_a, x_l, x_t = x_a.to(DEVICE), x_l.to(DEVICE), x_t.to(DEVICE)
        mu, _ = model.encode(x_a, x_l, x_t)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus)   # (N, latent_dim)


# ─────────────────────────────────────────────
# 6. 聚类
# ─────────────────────────────────────────────

def find_optimal_k(vectors, k_range=range(3, 11)):
    """用 Elbow + Silhouette 选最优 K（可传高维latent或2D UMAP坐标）"""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(vectors)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(vectors, labels,
                                            metric="cosine",
                                            sample_size=min(2000, len(vectors))))
        print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(list(k_range), inertias, "o-")
    ax1.set(title="Elbow Method", xlabel="k", ylabel="Inertia")
    ax2.plot(list(k_range), silhouettes, "o-", color="orange")
    ax2.set(title="Silhouette Score", xlabel="k", ylabel="Score")
    plt.tight_layout()
    plt.savefig("cluster_selection.png", dpi=150)
    plt.close()
    print("Saved: cluster_selection.png")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"Best k by silhouette: {best_k}")
    return best_k


def compute_umap(latent_vectors):
    """单独的UMAP降维，只跑一次"""
    print("Running UMAP (this may take ~1-2 min on CPU)...")
    pre = PCA(n_components=min(50, latent_vectors.shape[1]), random_state=RANDOM_SEED)
    latent_pca = pre.fit_transform(latent_vectors)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=10,   # 从30降到15，更关注局部结构
        min_dist=0.05,    # 从0.1降到0.05，点聚得更紧
        random_state=RANDOM_SEED
    )
    return reducer.fit_transform(latent_pca)


def run_clustering(vectors, n_clusters=N_CLUSTERS):
    """在传入的向量空间（高维latent或2D UMAP）上做KMeans聚类"""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    labels = km.fit_predict(vectors)
    score = silhouette_score(vectors, labels)
    print(f"KMeans k={n_clusters}: silhouette={score:.4f}")
    return labels, km


# ─────────────────────────────────────────────
# 7. 可视化
# ─────────────────────────────────────────────

def visualize_latent(latent_vectors, labels, df_clean, save_prefix="viz", coords_2d=None):
    """UMAP 2D 散点图 + 时间趋势堆叠图"""
    # ── 7.1 UMAP 投影 ────────────────────────────
    if coords_2d is None:
        print("Running UMAP (this may take ~1-2 min on CPU)...")
        pre = PCA(n_components=min(50, latent_vectors.shape[1]), random_state=RANDOM_SEED)
        latent_pca = pre.fit_transform(latent_vectors)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_SEED,
            verbose=False,
        )
        coords = reducer.fit_transform(latent_pca)
    else:
        coords = coords_2d  # 直接复用，不重新算

    # 离散颜色，cluster边界更清晰
    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", n_clusters)

    plt.figure(figsize=(8, 6))
    for k in range(n_clusters):
        mask = labels == k
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[cmap(k)], s=8, alpha=0.5, label=f"Cluster {k}")
    plt.legend(markerscale=2, loc="best")
    plt.title("VAE Latent Space (UMAP 2D)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_umap.png", dpi=150)
    plt.close()
    print(f"Saved: {save_prefix}_umap.png")

    # ── 7.2 按 Decade 的聚类比例 ────────────────
    df_plot = df_clean.copy()
    df_plot["cluster"] = labels
    df_plot = df_plot[df_plot["year"] > 0]
    df_plot["decade"] = (df_plot["year"] // 10 * 10).astype(int)

    pivot = (df_plot.groupby(["decade", "cluster"])
                    .size()
                    .unstack(fill_value=0))
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    pivot_norm.plot(kind="area", stacked=True, figsize=(10, 5),
                    colormap="tab10", alpha=0.8)
    plt.title("Cluster Distribution by Decade (Normalized)")
    plt.xlabel("Decade")
    plt.ylabel("Proportion")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_decade_trend.png", dpi=150)
    plt.close()
    print(f"Saved: {save_prefix}_decade_trend.png")

    return pivot, pivot_norm


def plot_training_history(history, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # 改成4格
    axes[0].plot(history["recon_a"], label="Acoustic")
    axes[0].set(title="Acoustic Recon Loss", xlabel="Epoch")
    axes[1].plot(history["recon_l"], label="Lyric", color="orange")
    axes[1].set(title="Lyric Recon Loss", xlabel="Epoch")
    axes[2].plot(history["recon_t"], label="Terms", color="green")
    axes[2].set(title="Terms Recon Loss", xlabel="Epoch")
    axes[3].plot(history["kl"], color="red")
    axes[3].set(title="KL Divergence", xlabel="Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 8. 主流程
# ─────────────────────────────────────────────

def main():
    # ── Step 1: 预处理 ────────────────────────────────
    print("\n" + "="*50)
    print("Step 1: Preprocessing")
    acoustic, lyric, terms, df_clean, acoustic_cols, scaler, tfidf = \
        load_and_preprocess(DATA_PATH)

    # has_lyrics flag（用于 loss masking）
    has_lyrics = (df_clean["lyrics_available"].values == 1).astype(np.float32)
    has_terms = (terms.sum(axis=1) > 0).astype(np.float32)

    # ── Step 2: 构建 DataLoader ───────────────────────
    dataset = TensorDataset(
        torch.tensor(acoustic, dtype=torch.float32),
        torch.tensor(lyric,    dtype=torch.float32),
        torch.tensor(terms,    dtype=torch.float32),   # 新增
        torch.tensor(has_lyrics, dtype=torch.float32),
        torch.tensor(has_terms,  dtype=torch.float32), # 新增
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE.type=="cuda"))

    # ── Step 3: 初始化 VAE ────────────────────────────
    print("\n" + "="*50)
    print("Step 2: Building VAE")
    acoustic_dim = acoustic.shape[1]
    lyric_dim = lyric.shape[1]
    terms_dim = terms.shape[1]
    model = MultimodalVAE(acoustic_dim, lyric_dim, terms_dim, LATENT_DIM)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")
    print(f"Acoustic dim: {acoustic_dim}, Lyric dim: {lyric_dim}, "
          f"Latent dim: {LATENT_DIM}")

    # ── Step 4: 训练 ──────────────────────────────────
    print("\n" + "="*50)
    print("Step 3: Training VAE")
    history = train_vae(model, loader, epochs=EPOCHS, lr=LR, beta_max=0.5)
    plot_training_history(history)

    # 保存模型权重
    torch.save(model.state_dict(), "vae_weights.pt")
    print("Saved: vae_weights.pt")

    # ── Step 5: 提取 Latent 向量 ─────────────────────
    print("\n" + "="*50)
    print("Step 4: Extracting Latent Vectors")
    full_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    latent_vectors = extract_latent(model, full_loader)
    print(f"Latent vectors shape: {latent_vectors.shape}")
    np.save("latent_vectors.npy", latent_vectors)

    # ── Step 6: 在 32D 高维空间上选 K & 聚类 ──────────────────────
    print("\n" + "="*50)
    print("Step 5: Clustering")
    # 【核心修改】：传入高维隐向量 latent_vectors，而不是 coords_2d
    print("Finding optimal k on 32D Latent Space...")
    best_k = find_optimal_k(latent_vectors, k_range=range(3, 11)) 

    print(f"\nRunning KMeans with k={best_k}...")
    # 【核心修改】：在高维空间运行聚类得到更细腻的标签
    labels, km = run_clustering(latent_vectors, n_clusters=best_k) 

    # ── Step 5.5: 计算 UMAP（仅用于可视化降维）─────────────
    print("Computing UMAP embedding for visualization...")
    coords_2d = compute_umap(latent_vectors)

    # ── Step 7: 可视化渲染 ────────────────────────────
    print("\n" + "="*50)
    print("Step 6: Visualization")
    # 把在 32D 聚类出来的丰富标签，和用于画图的 coords_2d 一起传进去
    pivot, pivot_norm = visualize_latent(latent_vectors, labels, df_clean, coords_2d=coords_2d)

    # 保存结果
    df_clean["cluster"] = labels
    df_clean.to_csv("msd_clustered.csv", index=False)
    print("Saved: msd_clustered.csv")

    # ── Step 8: 每个 Cluster 的特征摘要 ─────────────
    print("\n" + "="*50)
    print("Cluster Summary (acoustic means):")
    summary_cols = ["loudness", "tempo", "mode"] + \
                   [f"avg_timbre_{i}" for i in range(4)]
    summary = df_clean.groupby("cluster")[summary_cols].mean().round(3)
    print(summary.to_string())

    print("\nDecade distribution (counts):")
    df_clean["decade"] = (df_clean["year"].clip(lower=1) // 10 * 10).astype(int)
    print(df_clean.groupby(["decade", "cluster"]).size().unstack(fill_value=0))

    print("\nDone! 主要输出文件:")
    print("  msd_clustered.csv      — 含 cluster 标签的完整数据")
    print("  latent_vectors.npy     — VAE latent 向量 (N, 16)")
    print("  vae_weights.pt         — 模型权重")
    print("  training_history.png   — 训练曲线")
    print("  cluster_selection.png  — Elbow + Silhouette")
    print("  viz_pca.png            — Latent space PCA 散点图")
    print("  viz_decade_trend.png   — Cluster 时间趋势图")


if __name__ == "__main__":
    main()