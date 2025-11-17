import os
import re
from pathlib import Path
import warnings
from collections import OrderedDict
import torch
import esm
import biotite.structure.io as bsio

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore", message=".*libomp.dll.*")

FASTA_DIR = Path("fasta_files")   # name of your folder
FASTA_FILES = sorted([str(p) for p in FASTA_DIR.glob("*.fasta")])

OUTDIR = Path("tests/myoglobin_runs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Reduce VRAM
CHUNK = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta_one(path):
    """Read a single sequence from FASTA file (strip headers/whitespace)."""
    txt = Path(path).read_text().strip()
    seq = "".join(
        [line.strip() for line in txt.splitlines() if not line.startswith(">")]
    )
    seq = re.sub(r"\s+", "", seq)
    return seq

def write_text(p, s):
    """Write text to file, ensuring parent folder exists."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(s)

def calculate_plddt(pdb_path):
    """Mean pLDDT from the PDB B-factor column."""
    struct = bsio.load_structure(str(pdb_path), extra_fields=["b_factor"])
    return float(struct.b_factor.mean())

def main():
    print("Loading ESMFold...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    print(next(model.parameters()).device)
    model.set_chunk_size(CHUNK)
    print("ESMFold loaded successfully")

    records = []
    seqs = OrderedDict()

    # Predict each FASTA
    for fa in FASTA_FILES:
        name = Path(fa).stem
        seq = read_fasta_one(fa)
        seqs[name] = seq

        print(f"Folding {name} (len={len(seq)})...")
        with torch.no_grad():
            pdb_str = model.infer_pdb(seq)

        pdb_path = OUTDIR / f"{name}.pdb"
        write_text(pdb_path, pdb_str)
        print(f"  Saved to {pdb_path}")

        # Per-protein metric: mean pLDDT
        plddt_mean = calculate_plddt(pdb_path)
        records.append((name, len(seq), plddt_mean, str(pdb_path)))

    summary_path = OUTDIR / "summary.tsv"
    with open(summary_path, "w") as f:
        f.write("name\tlength\tpLDDT_mean\tpdb\n")
        for name, L, mean_plddt, pdbp in records:
            f.write(f"{name}\t{L}\t{mean_plddt:.2f}\t{pdbp}\n")

    print("\nSUMMARY:")
    for name, L, mean_plddt, pdbp in records:
        print(f"- {name}: len={L}, mean pLDDT≈{mean_plddt:.1f}, pdb={pdbp}")

    print(f"\nPer-protein summary written to: {summary_path}")

if __name__ == "__main__":
    main()
