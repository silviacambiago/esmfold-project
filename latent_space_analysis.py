import os
import re
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import esm


FASTA_DIR = Path("fasta_files")
PRED_DIR  = Path("tests/myoglobin_runs")
GT_DIR    = Path("ground_truths")

OUTDIR = Path("tests/latent_runs")
OUTDIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
ESM_MODEL = "esm2_t33_650M_UR50D"
ESM_LAYER = 33

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore", message=".*libomp.dll.*")

def read_fasta_seq(path: Path) -> str:
    txt = path.read_text().strip()
    seq = "".join([ln.strip() for ln in txt.splitlines() if not ln.startswith(">")])
    return re.sub(r"\s+", "", seq).upper()

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def save_matrix_tsv(path: Path, labels: List[str], mat: np.ndarray, header: str):
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write("name\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            row = "\t".join(f"{x:.6f}" for x in mat[i])
            f.write(f"{lab}\t{row}\n")

def load_esm2(model_name=ESM_MODEL, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_load_fn = getattr(esm.pretrained, model_name)
    model, alphabet = model_load_fn()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, device

def embed_sequences_esm2(items, model, alphabet, batch_converter, device, layer=ESM_LAYER):
    seq_tuples = []
    for label, path in items:
        seq = read_fasta_seq(path)
        seq_tuples.append((label, seq))

    name_to_vec = {}

    with torch.no_grad():
        for chunk in batched(seq_tuples, BATCH_SIZE):
            labels, strs, toks = batch_converter(chunk)
            toks = toks.to(device)

            out = model(toks, repr_layers=[layer], return_contacts=False)
            reps = out["representations"][layer]

            for i, (label, seq) in enumerate(chunk):
                rep = reps[i, 1:1+len(seq), :]
                vec = rep.mean(dim=0).cpu().numpy()
                name_to_vec[label] = vec

    return name_to_vec

def pairwise_cosine(X):
    eps = 1e-12
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, eps)
    return Xn @ Xn.T

def pairwise_euclidean(X):
    s = np.sum(X**2, axis=1, keepdims=True)
    D2 = s + s.T - 2 * (X @ X.T)
    D2[D2 < 0] = 0.0
    return np.sqrt(D2)

def pca_reduce(X, n=2):
    return PCA(n_components=n).fit_transform(X)

def scatter_plot(Z, names, groups, out_png, title):
    import numpy as np
    plt.figure(figsize=(7, 6), dpi=140)

    xs, ys = Z[:, 0], Z[:, 1]

    style = {
        "pred": {"marker": "o", "color": "blue",  "label": "Predicted"},
        "gt":   {"marker": "s", "color": "red",   "label": "Ground Truth"},
    }

    jitter_scale = 0.03

    for grp in ["gt", "pred"]:
        idx = [i for i, g in enumerate(groups) if g == grp]
        if not idx:
            continue

        xs_g = xs[idx].copy()
        ys_g = ys[idx].copy()

        if grp == "pred":
            rng = np.random.default_rng(0)
            xs_g += rng.normal(scale=jitter_scale, size=len(xs_g))
            ys_g += rng.normal(scale=jitter_scale, size=len(ys_g))

        plt.scatter(xs_g, ys_g,
                    color=style[grp]["color"],
                    marker=style[grp]["marker"],
                    label=style[grp]["label"],
                    s=60)

        for x, y, lab in zip(xs_g, ys_g, [names[i] for i in idx]):
            plt.text(x, y, lab, fontsize=7)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():

    fasta_files = sorted(FASTA_DIR.glob("*.fasta"))
    if not fasta_files:
        raise RuntimeError("fasta_files/ is empty!")

    items = []
    names = []
    groups = []

    for fa in fasta_files:
        stem = fa.stem

        pred_label = f"{stem}_pred"
        gt_label   = f"{stem}_gt"

        items.append((pred_label, fa))
        items.append((gt_label, fa))

        names.append(stem)
        groups.append("pred")

        names.append(stem)
        groups.append("gt")

    print("Loading ESM-2…")
    model, alphabet, batch_converter, device = load_esm2()
    print(f"Device: {device}")

    print("Embedding sequences…")
    name_to_vec = embed_sequences_esm2(items, model, alphabet, batch_converter, device)

    labels_in_order = [lbl for (lbl,_) in items]
    X = np.stack([name_to_vec[lbl] for lbl in labels_in_order])
    np.save(OUTDIR / "embeddings.npy", X)

    print("Computing distances…")
    C = pairwise_cosine(X)
    D = pairwise_euclidean(X)

    matrix_labels = [f"{nm}({grp})" for nm,grp in zip(names,groups)]

    save_matrix_tsv(OUTDIR / "cosine_similarity.tsv", matrix_labels, C, "cosine_similarity")
    save_matrix_tsv(OUTDIR / "euclidean_distance.tsv", matrix_labels, D, "euclidean_distance")

    print("Running PCA…")
    Zp = pca_reduce(X, 2)
    scatter_plot(Zp, names, groups, OUTDIR / "latent_pca.png",
                 "ESMFold latent space (PCA): Predicted vs Ground Truth")

    print("\nDone.")
    print(f"Saved PCA plot: {OUTDIR/'latent_pca.png'}")

if __name__ == "__main__":
    main()
