import torch
import esm
import biotite.structure.io as bsio

# -------------------------------
# Device selection
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------------------
# Load ESMFold v1
# -------------------------------
model = esm.pretrained.esmfold_v1()

# ↓↓↓ IMPORTANT: reduce memory usage ↓↓↓
# Smaller chunk sizes => lower peak memory, slower runtime
model.set_chunk_size(64)   # try 64 first; if it still gets killed, try 32
# ↑↑↑ IMPORTANT ↑↑↑

model = model.eval().to(DEVICE)

# -------------------------------
# Example sequence
# -------------------------------
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# For multimers: "CHAIN1:CHAIN2" etc.

# -------------------------------
# Inference
# -------------------------------
with torch.no_grad():
    output = model.infer_pdb(sequence)

# -------------------------------
# Save PDB and compute mean pLDDT
# -------------------------------
pdb_path = "result.pdb"
with open(pdb_path, "w") as f:
    f.write(output)

struct = bsio.load_structure(pdb_path, extra_fields=["b_factor"])
print("Mean pLDDT:", struct.b_factor.mean())
