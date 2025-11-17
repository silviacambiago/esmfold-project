"""
ESM2 scoring for wild-type and mutant proteins.

Pipeline:
- Reads all FASTA files from fasta_files/.
- Uses filename convention to group WT and mutants:
    * WT:   <base_id>_wt.fasta
    * MUT:  <base_id>_mutX.fasta (anything with same base_id and not '_wt')
- For each sequence (WT and mutants), computes:
    - pseudo log-likelihood (PLL): sum_i log p(x_i | x_{-i}) via masking
    - avg_log_prob = PLL / L
    - pseudo-perplexity = exp(-avg_log_prob)
    - an ESM2 embedding (mean over residues)
- For each mutant, also computes mutation LLR vs the WT sequence:
    - LLR = sum over mutated positions i of
        [ log p(mut_aa_i | WT context, position i masked)
        - log p(wt_aa_i  | WT context, position i masked) ]
- Writes:
    - tests/esm2_scores/esm2_scores.tsv (one row per variant)
    - tests/esm2_scores/embeddings/<variant_id>.npy
"""
import os
import re
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import esm


FASTA_DIR = Path("esm2_fasta")
OUTDIR = Path("tests/esm2_scores")
EMB_DIR = OUTDIR / "embeddings"

OUTDIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)

ESM_MODEL = "esm2_t33_650M_UR50D"
BATCH_SIZE_EMB = 4   # for embeddings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*libomp.dll.*")


def read_fasta_seq(path: Path) -> str:
    txt = path.read_text().strip()
    seq = "".join(
        ln.strip() for ln in txt.splitlines()
        if not ln.startswith(">")
    )
    seq = re.sub(r"\s+", "", seq).upper()
    return seq

def list_fasta_files(fasta_dir: Path) -> List[Path]:
    return sorted(list(fasta_dir.glob("*.fasta")))


def parse_variant_name(stem: str) -> Tuple[str, str, bool]:
    """
    Given a fasta stem (filename without .fasta), return:
        base_id, variant_id, is_wt

    Convention:
      - WT: stem ends with '_wt' -> base_id = stem[:-3]
      - MUT:
          if '_mut' in stem: base_id = stem.split('_mut')[0]
          else: base_id = stem.rsplit('_', 1)[0] (fallback)
    """
    if stem.endswith("_wt"):
        base_id = stem[:-3]
        return base_id, stem, True

    # mutant
    if "_mut" in stem:
        base_id = stem.split("_mut")[0]
    else:
        if "_" in stem:
            base_id = stem.rsplit("_", 1)[0]
        else:
            base_id = stem
    return base_id, stem, False


def load_esm2(model_name: str = ESM_MODEL, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_load_fn = getattr(esm.pretrained, model_name)
    model, alphabet = model_load_fn()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    return model, alphabet, batch_converter, device, mask_idx


def pseudo_log_likelihood(
    seq: str,
    model,
    alphabet,
    batch_converter,
    device: str,
    mask_idx: int,
    pos_batch_size: int = 16,
) -> Tuple[float, float, float]:
    """
    Compute pseudo-log-likelihood (PLL) and pseudo-perplexity.

    PLL = sum_i log p(x_i | x_{-i}) where x_i is masked one position at a time.
    avg_log_prob = PLL / L
    pseudo_perplexity = exp(-avg_log_prob)

    To avoid CUDA OOM on long sequences, we do this in chunks of positions
    (pos_batch_size) instead of masking all L positions at once.
    """
    if not seq:
        return float("nan"), float("nan"), float("nan")

    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)  # [1, T]
    tokens = tokens.to(device)
    L = len(seq)

    pll = 0.0

    for start in range(0, L, pos_batch_size):
        end = min(L, start + pos_batch_size)
        chunk_positions = list(range(start, end))
        bsz = len(chunk_positions)

        batch_tokens = tokens.repeat(bsz, 1)

        for row_idx, pos in enumerate(chunk_positions):
            pos_idx = pos + 1
            batch_tokens[row_idx, pos_idx] = mask_idx

        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[], return_contacts=False)
            logits = out["logits"]

        log_softmax = torch.log_softmax(logits, dim=-1)

        for row_idx, pos in enumerate(chunk_positions):
            pos_idx = pos + 1
            true_idx = tokens[0, pos_idx]
            logp = log_softmax[row_idx, pos_idx, true_idx].item()
            pll += logp

    avg_log_prob = pll / L
    pseudo_pp = math.exp(-avg_log_prob)
    return float(pll), float(avg_log_prob), float(pseudo_pp)


def mutation_llr_vs_wt(
    wt_seq: str,
    mut_seq: str,
    model,
    alphabet,
    batch_converter,
    device: str,
    mask_idx: int
) -> Tuple[float, List[int]]:
    """
    Compute LLR(mut vs WT) using masked marginal scoring.

    For each position i where wt_seq[i] != mut_seq[i]:
        - mask position i in the WT background
        - compute log p(wt_aa | context) and log p(mut_aa | context)
        - add log p(mut_aa) - log p(wt_aa) to LLR

    Returns:
        (LLR, mutated_positions)
    """
    if len(wt_seq) != len(mut_seq):
        return float("nan"), []

    diff_positions = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
    if not diff_positions:
        return 0.0, []

    data = [("wt", wt_seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    batch_tokens = tokens.repeat(len(diff_positions), 1)
    for row_idx, pos in enumerate(diff_positions):
        pos_idx = pos + 1
        batch_tokens[row_idx, pos_idx] = mask_idx

    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[], return_contacts=False)
        logits = out["logits"]  # [K, T, vocab]
    log_softmax = torch.log_softmax(logits, dim=-1)

    llr = 0.0
    for row_idx, pos in enumerate(diff_positions):
        pos_idx = pos + 1
        wt_aa = wt_seq[pos]
        mut_aa = mut_seq[pos]

        wt_idx = alphabet.get_idx(wt_aa)
        mut_idx = alphabet.get_idx(mut_aa)

        logp_wt = log_softmax[row_idx, pos_idx, wt_idx].item()
        logp_mut = log_softmax[row_idx, pos_idx, mut_idx].item()
        llr += (logp_mut - logp_wt)

    return float(llr), diff_positions


def embed_sequences(
    seq_items: List[Tuple[str, str]],
    model,
    alphabet,
    batch_converter,
    device: str
) -> Dict[str, np.ndarray]:
    """
    seq_items: list of (variant_id, seq)
    Returns dict {variant_id: embedding}, where embedding = mean over residue representations.
    """
    name_to_vec: Dict[str, np.ndarray] = {}

    if not seq_items:
        return name_to_vec

    with torch.no_grad():
        for i in range(0, len(seq_items), BATCH_SIZE_EMB):
            chunk = seq_items[i:i + BATCH_SIZE_EMB]
            if not chunk:
                continue

            batch_data = [(vid, seq) for vid, seq in chunk]
            _, _, toks = batch_converter(batch_data)
            toks = toks.to(device)

            layer = 33
            out = model(
                toks,
                repr_layers=[layer],
                return_contacts=False
            )
            reps = out["representations"][layer]

            for row_idx, (vid, seq) in enumerate(chunk):
                L = len(seq)
                rep = reps[row_idx, 1:1+L, :]
                vec = rep.mean(dim=0).cpu().numpy()
                name_to_vec[vid] = vec

    return name_to_vec


def main():
    fasta_files = list_fasta_files(FASTA_DIR)
    if not fasta_files:
        raise RuntimeError(f"No FASTA files found in {FASTA_DIR!s}")

    wt_map: Dict[str, Tuple[str, str]] = {}
    mut_map: Dict[str, List[Tuple[str, str]]] = {}

    for fa in fasta_files:
        stem = fa.stem
        base_id, variant_id, is_wt = parse_variant_name(stem)
        seq = read_fasta_seq(fa)

        if is_wt:
            wt_map[base_id] = (variant_id, seq)
        else:
            mut_map.setdefault(base_id, []).append((variant_id, seq))

    print(f"Found {len(wt_map)} WT base_ids and {sum(len(v) for v in mut_map.values())} mutants.")

    print("Loading ESM2 model...")
    model, alphabet, batch_converter, device, mask_idx = load_esm2()
    print(f"Model: {ESM_MODEL} | Device: {device}")

    seq_items: List[Tuple[str, str]] = []

    for base_id, (vid_wt, seq_wt) in wt_map.items():
        seq_items.append((vid_wt, seq_wt))
        for vid_mut, seq_mut in mut_map.get(base_id, []):
            seq_items.append((vid_mut, seq_mut))

    for base_id, mut_list in mut_map.items():
        if base_id in wt_map:
            continue
        for vid_mut, seq_mut in mut_list:
            seq_items.append((vid_mut, seq_mut))

    print("Computing embeddings for all variants...")
    emb_dict = embed_sequences(seq_items, model, alphabet, batch_converter, device)

    for vid, vec in emb_dict.items():
        np.save(EMB_DIR / f"{vid}.npy", vec)

    rows = []
    print("Computing PLL / PPL and mutation LLRs...")

    for base_id, (wt_vid, wt_seq) in wt_map.items():
        pll_wt, avglog_wt, ppl_wt = pseudo_log_likelihood(
            wt_seq, model, alphabet, batch_converter, device, mask_idx
        )
        rows.append({
            "base_id": base_id,
            "variant_id": wt_vid,
            "is_wt": 1,
            "length": len(wt_seq),
            "log_pseudo_likelihood": pll_wt,
            "avg_log_prob": avglog_wt,
            "pseudo_perplexity": ppl_wt,
            "n_mut_positions": 0,
            "llr_vs_wt": float("nan"),
            "mut_positions": ""
        })

        for mut_vid, mut_seq in mut_map.get(base_id, []):
            pll_mut, avglog_mut, ppl_mut = pseudo_log_likelihood(
                mut_seq, model, alphabet, batch_converter, device, mask_idx
            )
            llr, mut_positions = mutation_llr_vs_wt(
                wt_seq, mut_seq, model, alphabet, batch_converter, device, mask_idx
            )
            rows.append({
                "base_id": base_id,
                "variant_id": mut_vid,
                "is_wt": 0,
                "length": len(mut_seq),
                "log_pseudo_likelihood": pll_mut,
                "avg_log_prob": avglog_mut,
                "pseudo_perplexity": ppl_mut,
                "n_mut_positions": len(mut_positions),
                "llr_vs_wt": llr,
                "mut_positions": ",".join(str(i) for i in mut_positions)
            })

    for base_id, mut_list in mut_map.items():
        if base_id in wt_map:
            continue
        for mut_vid, mut_seq in mut_list:
            pll_mut, avglog_mut, ppl_mut = pseudo_log_likelihood(
                mut_seq, model, alphabet, batch_converter, device, mask_idx
            )
            rows.append({
                "base_id": base_id,
                "variant_id": mut_vid,
                "is_wt": 0,
                "length": len(mut_seq),
                "log_pseudo_likelihood": pll_mut,
                "avg_log_prob": avglog_mut,
                "pseudo_perplexity": ppl_mut,
                "n_mut_positions": 0,
                "llr_vs_wt": float("nan"),
                "mut_positions": ""
            })

    out_tsv = OUTDIR / "esm2_scores.tsv"
    with open(out_tsv, "w") as f:
        header = [
            "base_id", "variant_id", "is_wt", "length",
            "log_pseudo_likelihood", "avg_log_prob", "pseudo_perplexity",
            "n_mut_positions", "llr_vs_wt", "mut_positions"
        ]
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[h]) for h in header) + "\n")

    print(f"\nDone. Scores written to: {out_tsv}")
    print(f"Embeddings saved to: {EMB_DIR}")

if __name__ == "__main__":
    main()
