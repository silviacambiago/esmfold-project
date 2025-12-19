#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==============================
# USER CONFIG (EDIT HERE ONLY)
# ==============================
NPZ_PATH = "esm2_ubiquitin_multi_embeddings.npz"   # output of compute_esm2_embeddings.py
FUNCTIONAL_QUANTILE = 0.95                 # quantile for functional region radius
MAX_HAMMING_FOR_FIT = 2                    # use functional seqs with Hamming <= this to fit region
OUT_PREFIX = "ubi_multi_space"               # prefix for figure + summary files
# ==============================


def mahalanobis_distances(X, mu, Sigma):
    """
    Compute Mahalanobis distance of rows of X from mean mu, covariance Sigma.

    X: (N, 3)
    mu: (3,)
    Sigma: (3, 3)
    """
    Xc = X - mu
    Sigma_inv = np.linalg.inv(Sigma)
    left = Xc @ Sigma_inv
    d2 = np.sum(left * Xc, axis=1)
    return np.sqrt(d2)


def sample_ellipsoid(mu, Sigma, radius, n_u=40, n_v=20):
    """
    Sample points on the ellipsoidal boundary defined by
        (x - mu)^T Sigma^{-1} (x - mu) = radius^2
    in 3D, and return (X, Y, Z) grids.
    """
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    u, v = np.meshgrid(u, v)

    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    sphere = np.stack([xs, ys, zs], axis=-1)  # (n_v, n_u, 3)

    # Cholesky factor of Sigma
    L = np.linalg.cholesky(Sigma)

    ellip = sphere @ L.T
    ellip *= radius
    ellip += mu

    X = ellip[..., 0]
    Y = ellip[..., 1]
    Z = ellip[..., 2]
    return X, Y, Z


def main():
    print(f"Loading embeddings from: {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)

    embeddings = data["embeddings"]        # (N, D)
    esm_scores = data["esm_scores"]        # (N,)
    fitness = data["fitness"]              # (N,)  (ESM-derived score)
    labels = data["labels"]                # (N,)  0/1 functional label
    hamdist = data["hamming_distance"]     # (N,)
    seq_ids = data["seq_ids"]              # (N,)

    print("Loaded", embeddings.shape[0], "sequences with dim", embeddings.shape[1])

    # ---------------- PCA to 3D ----------------
    print("Running PCA to 3D...")
    pca = PCA(n_components=3)
    X3 = pca.fit_transform(embeddings)  # (N, 3)

    print("Explained variance by first 3 PCs:", pca.explained_variance_ratio_)

    # ---------------- Fit functional region (ellipsoid) ----------------
    mask_fit = (labels == 1) & (hamdist <= MAX_HAMMING_FOR_FIT)
    X_fit = X3[mask_fit]

    if X_fit.shape[0] < 5:
        raise RuntimeError(
            f"Not enough functional sequences within Hamming <= {MAX_HAMMING_FOR_FIT} "
            f"to fit functional region (only {X_fit.shape[0]} sequences)."
        )

    mu = X_fit.mean(axis=0)
    Sigma = np.cov(X_fit.T)  # (3, 3)

    d_fit = mahalanobis_distances(X_fit, mu, Sigma)
    R = np.quantile(d_fit, FUNCTIONAL_QUANTILE)

    print(f"Functional region radius (Mahalanobis, quantile {FUNCTIONAL_QUANTILE}): {R:.4f}")

    d_all = mahalanobis_distances(X3, mu, Sigma)
    inside_region = d_all <= R

    # ---------------- Basic counts ----------------
    is_func = (labels == 1)
    is_nonfunc = (labels == 0)

    print("\nCounts:")
    print("  Total sequences:", len(labels))
    print("  Functional:", int(is_func.sum()))
    print("  Non-functional:", int(is_nonfunc.sum()))
    print("  Inside functional region:", int(inside_region.sum()))
    print("  Outside functional region:", int((~inside_region).sum()))

    # ---------------- Define mutation-step categories ----------------
    ham0 = (hamdist == 0)
    ham1 = (hamdist == 1)
    ham2 = (hamdist == 2)
    ham_ge3 = (hamdist >= 3)

    func_h1 = is_func & ham1
    func_h2 = is_func & ham2
    nonfunc_h1 = is_nonfunc & ham1
    nonfunc_h2 = is_nonfunc & ham2

    func_other = is_func & ham_ge3
    nonfunc_other = is_nonfunc & ham_ge3

    # ---------------- 3D scatter plot ----------------
    print("Plotting 3D latent space with functional region...")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    def scatter_cat(mask, color, label, alpha=0.85, size=28, marker="o"):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        ax.scatter(
            X3[idx, 0],
            X3[idx, 1],
            X3[idx, 2],
            c=color,
            label=f"{label} (n={len(idx)})",
            alpha=alpha,
            s=size,
            marker=marker,
        )

    # WT
    if ham0.any():
        idx_wt = np.where(ham0)[0][0]
        ax.scatter(
            X3[idx_wt, 0],
            X3[idx_wt, 1],
            X3[idx_wt, 2],
            c="k",
            marker="*",
            s=160,
            label="WT",
        )

    # Functional mutants
    scatter_cat(func_h1, "tab:blue", "Functional 1-step mutants")
    scatter_cat(func_h2, "tab:green", "Functional 2-step mutants")

    # Non-functional mutants
    scatter_cat(nonfunc_h1, "tab:orange", "Non-functional 1-step mutants")
    scatter_cat(nonfunc_h2, "tab:red", "Non-functional 2-step mutants")
    scatter_cat(func_other, "tab:cyan", "Functional (≥3 steps)", alpha=0.6, size=20)
    scatter_cat(nonfunc_other, "tab:gray", "Non-functional (≥3 steps)", alpha=0.6, size=20)

    # ---------------- Functional boundary as a single smooth shell ----------------
    Xe, Ye, Ze = sample_ellipsoid(mu, Sigma, R, n_u=40, n_v=20)

    # light, semi-transparent surface (the "bubble" around functional sequences)
    ax.plot_surface(
        Xe, Ye, Ze,
        rstride=2,
        cstride=2,
        color="lightgray",
        alpha=0.18,
        linewidth=0,
        edgecolor="none",
    )
    # optional faint wireframe just to suggest shape
    ax.plot_wireframe(
        Xe, Ye, Ze,
        rstride=6,
        cstride=6,
        color="gray",
        linewidth=0.4,
        alpha=0.25,
        label="Functional region boundary",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("ESM2 latent space (PCA 3D) with functional region")

    # choose a gentle view like in your sketch
    ax.view_init(elev=22, azim=35)

    ax.legend(loc="best")

    out_png = Path(f"{OUT_PREFIX}_3d_space.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("Saved 3D plot to", out_png)

    # ---------------- Summary table ----------------
    out_csv = Path(f"{OUT_PREFIX}_summary.csv")

    summary_df = pd.DataFrame({
        "seq_id": seq_ids,
        "fitness": fitness,
        "label_functional": labels,
        "hamming_distance": hamdist,
        "pc1": X3[:, 0],
        "pc2": X3[:, 1],
        "pc3": X3[:, 2],
        "mahalanobis_d": d_all,
        "inside_functional_region": inside_region.astype(int),
        "esm_score": esm_scores,
    })

    summary_df["mutation_step_category"] = np.select(
        [ham0, ham1, ham2, ham_ge3],
        ["wt", "1_step", "2_step", ">=3_step"],
        default="unknown",
    )

    summary_df.to_csv(out_csv, index=False)
    print("Saved summary table to", out_csv)


if __name__ == "__main__":
    main()
