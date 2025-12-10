"""
Fold all FASTA files in a directory with ESMFold and write:
- one PDB per FASTA
- a summary.tsv with name, length, mean pLDDT, and PDB path

Designed to run on an HPC cluster (e.g. in a Slurm job).
"""

import os
import re
import warnings
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import esm
import biotite.structure.io as bsio

# Optional: avoid OpenMP spam
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", message=".*libomp.dll.*")


def read_fasta_one(path: Path) -> str:
    """Read a single sequence from FASTA file (strip headers/whitespace)."""
    txt = path.read_text().strip()
    seq = "".join(
        line.strip()
        for line in txt.splitlines()
        if not line.startswith(">")
    )
    seq = re.sub(r"\s+", "", seq)
    return seq


def write_text(path: Path, s: str) -> None:
    """Write text to file, ensuring parent folder exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(s)


def calculate_plddt(pdb_path: Path) -> float:
    """Mean pLDDT from the PDB B-factor column."""
    struct = bsio.load_structure(str(pdb_path), extra_fields=["b_factor"])
    return float(struct.b_factor.mean())


def main():
    parser = argparse.ArgumentParser(description="Batch ESMFold on all FASTA files in a directory.")
    parser.add_argument(
        "--fasta-dir",
        type=str,
        default="fasta_files",
        help="Directory containing input .fasta files (default: fasta_files)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="tests/myoglobin_runs",
        help="Output directory for PDBs and summary.tsv (default: tests/myoglobin_runs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="ESMFold chunk size to reduce VRAM usage (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu' (default: cuda)",
    )
    args = parser.parse_args()

    fasta_dir = Path(args.fasta_dir)
    outdir = Path(args.outdir)
    chunk_size = args.chunk_size

    if not fasta_dir.is_dir():
        raise FileNotFoundError(f"FASTA directory not found: {fasta_dir}")

    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        raise RuntimeError(f"No .fasta files found in {fasta_dir}")

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    # Optional: respect OMP_NUM_THREADS if set by the scheduler
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass

    # Load ESMFold
    print("[INFO] Loading ESMFold model...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    model.set_chunk_size(chunk_size)
    print(f"[INFO] ESMFold loaded on device: {next(model.parameters()).device}")
    print(f"[INFO] Chunk size set to: {chunk_size}")

    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    seqs = OrderedDict()

    # Predict each FASTA
    for fa in fasta_files:
        name = fa.stem
        seq = read_fasta_one(fa)
        seqs[name] = seq

        print(f"[INFO] Folding {name} (len={len(seq)})...")
        with torch.no_grad():
            pdb_str = model.infer_pdb(seq)

        pdb_path = outdir / f"{name}.pdb"
        write_text(pdb_path, pdb_str)
        print(f"[INFO]   Saved PDB to {pdb_path}")

        # Per-protein metric: mean pLDDT
        plddt_mean = calculate_plddt(pdb_path)
        records.append((name, len(seq), plddt_mean, str(pdb_path)))

    summary_path = outdir / "summary.tsv"
    with open(summary_path, "w") as f:
        f.write("name\tlength\tpLDDT_mean\tpdb\n")
        for name, L, mean_plddt, pdbp in records:
            f.write(f"{name}\t{L}\t{mean_plddt:.2f}\t{pdbp}\n")

    print("\n[SUMMARY]")
    for name, L, mean_plddt, pdbp in records:
        print(f"  - {name}: len={L}, mean pLDDT≈{mean_plddt:.1f}, pdb={pdbp}")

    print(f"\n[INFO] Per-protein summary written to: {summary_path}")


if __name__ == "__main__":
    main()