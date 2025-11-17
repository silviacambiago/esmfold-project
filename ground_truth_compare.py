import re
from pathlib import Path
import numpy as np
from typing import Union, Tuple, List, Dict
from Bio.PDB import PDBParser, Superimposer
from Bio import pairwise2

FASTA_DIR = Path("fasta_files")
PRED_DIR  = Path("tests/myoglobin_runs")
GT_DIR    = Path("ground_truths")

# 3-letter -> 1-letter map (extend if needed)
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"C","PYL":"K"
}

def _pdb_seqres_sequences(pdb_path: Union[str, Path]) -> Dict[str, str]:
    """
    Parse SEQRES records from a PDB (legacy .pdb) file.
    Returns {chain_id: one_letter_seq}. If no SEQRES present, returns {}.
    """
    chain_to_aa3: Dict[str, List[str]] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("SEQRES"):
                continue
            chain_id = line[11].strip()
            # residue names at cols (per PDB format) 20-70, groups of up to 13
            parts = line[19:].split()
            if chain_id not in chain_to_aa3:
                chain_to_aa3[chain_id] = []
            chain_to_aa3[chain_id].extend(parts)

    out = {}
    for cid, aa3_list in chain_to_aa3.items():
        one = "".join(AA3_TO_1.get(r.upper(), "X") for r in aa3_list)
        out[cid] = one
    return out

def _pdb_atom_sequence(pdb_path: Union[str, Path], chain_id: str) -> str:
    """
    Sequence from ATOM records (observed residues only).
    """
    s = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
    m = next(s.get_models())
    ch = m[chain_id]
    one = []
    seen_res_ids = set()
    for res in ch:
        # skip HETATM / waters / ligands
        if res.id[0] != " ":
            continue
        # avoid double-counting insertion codes/altlocs
        rid = (res.id[1], res.id[2])
        if rid in seen_res_ids:
            continue
        seen_res_ids.add(rid)
        rn = res.get_resname().upper()
        one.append(AA3_TO_1.get(rn, "X"))
    return "".join(one)

def _all_chain_ids(pdb_path: Union[str, Path]) -> List[str]:
    s = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
    m = next(s.get_models())
    return [ch.id for ch in m.get_chains()]

def _align_stats(q: str, t: str):
    """
    Align q (FASTA) to t (target chain seq) and compute:
    identity_pct over aligned non-gaps, coverage over min(len(q), len(t)),
    plus alignment strings.
    Uses a slightly stricter scoring than globalxx to discourage silly matches.
    """
    if not q or not t:
        return None
    aln = pairwise2.align.globalms(q, t, 2, -1, -5, -1, one_alignment_only=True)
    if not aln:
        return None
    a = aln[0]
    fa, fb = a.seqA, a.seqB
    aligned_pairs = [(x, y) for x, y in zip(fa, fb) if x != "-" and y != "-"]
    matches = sum(1 for x, y in aligned_pairs if x == y)
    aln_len = max(1, len(aligned_pairs))
    ident_pct = 100.0 * matches / aln_len
    coverage = len(aligned_pairs) / max(1, min(len(q), len(t)))
    return ident_pct, coverage, fa, fb

def best_chain_by_identity(
    fasta_seq: str,
    pdb_path: Union[str, Path],
    min_coverage: float = 0.75,
    min_identity: float = 30.0
) -> Tuple[str, float, Tuple[str, str, str]]:
    """
    Robust selector:
      1) Try SEQRES (full chain) for each chain; compute identity & coverage.
      2) If no SEQRES or poor coverage, fall back to ATOM-derived sequence.
      3) Pick chain maximizing a composite score: ident_pct * coverage.
      4) Enforce minimum coverage & identity; otherwise return best anyway.

    Returns (chain_id, identity_pct, (aln_fasta, aln_pdb, "")).
    """
    q = re.sub(r"\s+", "", fasta_seq.upper())

    # Gather candidate sequences
    seqres_map = _pdb_seqres_sequences(pdb_path)  # may be {}
    chain_ids = _all_chain_ids(pdb_path)

    best = ("", -1.0, ("", "", ""))
    best_score = -1.0
    best_passes = False

    for cid in chain_ids:
        # Prefer SEQRES if available; else ATOM
        candidates = []
        if cid in seqres_map and seqres_map[cid]:
            candidates.append(("SEQRES", seqres_map[cid]))
        try:
            atom_seq = _pdb_atom_sequence(pdb_path, cid)
            if atom_seq:
                candidates.append(("ATOM", atom_seq))
        except Exception:
            pass

        for source, target_seq in candidates:
            stats = _align_stats(q, target_seq)
            if not stats:
                continue
            ident_pct, cov, fa, fb = stats
            score = ident_pct * cov  # composite

            passes = (ident_pct >= min_identity and cov >= min_coverage)
            # prefer passing candidates; otherwise keep best overall
            better = False
            if passes and not best_passes:
                better = True
            elif passes == best_passes and score > best_score:
                better = True

            if better:
                best = (cid, ident_pct, (fa, fb, ""))
                best_score = score
                best_passes = passes

    return best


def coords_from_alignment(pred_pdb: Union[str, Path], gt_pdb: Union[str, Path],
                          chain_pred: str, chain_gt: str,
                          aln_pred: str, aln_gt: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return matched CA coords using the alignment (skip gaps)."""
    def ca_map(pdb_path, chain_id):
        s = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
        m = next(s.get_models())
        ch = m[chain_id]
        out: Dict[int, np.ndarray] = {}
        seq_idx = -1
        for res in ch:
            if res.id[0] != " ":
                continue
            if "CA" in res:
                seq_idx += 1
                out[seq_idx] = res["CA"].get_coord()
            else:
                seq_idx += 1
        return out

    pred_map = ca_map(pred_pdb, chain_pred)
    gt_map   = ca_map(gt_pdb, chain_gt)

    pred_xyz, gt_xyz = [], []
    i_pred = i_gt = -1
    for a, b in zip(aln_pred, aln_gt):
        if a != "-":
            i_pred += 1
        if b != "-":
            i_gt += 1
        if a != "-" and b != "-":
            if i_pred in pred_map and i_gt in gt_map:
                pred_xyz.append(pred_map[i_pred])
                gt_xyz.append(gt_map[i_gt])

    return np.array(pred_xyz, dtype=float), np.array(gt_xyz, dtype=float)

def super_rmsd(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    sup = Superimposer()
    # Build pseudo-Atom objects is heavy; Superimposer also accepts raw arrays via a hack:
    # Compute RMSD directly after Kabsch with numpy (simple & stable).
    # Center
    P = pred_xyz - pred_xyz.mean(0)
    Q = gt_xyz   - gt_xyz.mean(0)
    # Kabsch
    C = P.T @ Q
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1,1,d]) @ Wt
    P_aln = P @ R
    rmsd = float(np.sqrt(np.mean(np.sum((P_aln - Q)**2, axis=1))))
    return rmsd, R, gt_xyz.mean(0)

def tm_score(pred_xyz_aln: np.ndarray, gt_xyz: np.ndarray) -> float:
    n = len(gt_xyz)
    if n == 0:
        return float("nan")
    base = max(1.0, n - 15)
    d0 = max(0.5, 1.24 * np.cbrt(base) - 1.8)
    d = np.linalg.norm(pred_xyz_aln - gt_xyz, axis=1)
    return float(np.mean(1.0 / (1.0 + (d/d0)**2)))

def compare_prediction_to_pdb(fasta_seq: str,
                              predicted_pdb: Union[str, Path],
                              ground_truth_pdb: Union[str, Path]):
    # choose best GT chain
    gt_chain, id_pct, (aln_fa, aln_gt, _) = best_chain_by_identity(fasta_seq, ground_truth_pdb)

    if not gt_chain or id_pct < 50.0:
        return {"ok": False, "reason": f"No good chain match in GT PDB (best identity {id_pct:.1f}%)."}

    # predicted is single chain "A" from ESMFold
    pred_chain = "A"

    # build coords by alignment
    pred_xyz, gt_xyz = coords_from_alignment(predicted_pdb, ground_truth_pdb,
                                             pred_chain, gt_chain, aln_fa, aln_gt)
    if len(pred_xyz) < 20:
        return {"ok": False, "reason": f"Too few matched residues after alignment: {len(pred_xyz)}."}

    # superimpose & metrics
    rmsd, R, t_gt = super_rmsd(pred_xyz, gt_xyz)
    pred_aln = (pred_xyz - pred_xyz.mean(0)) @ R + t_gt
    tm = tm_score(pred_aln, gt_xyz)

    return {
        "ok": True,
        "gt_chain": gt_chain,
        "seq_identity_pct": id_pct,
        "n_aligned": int(len(gt_xyz)),
        "CA_RMSD_A": float(rmsd),
        "TM_score": float(tm),
    }

ENTRIES = []

for fasta_path in sorted(FASTA_DIR.glob("*.fasta")):
    stem = fasta_path.stem  # e.g. "myoglobin"

    pred_pdb = PRED_DIR / f"{stem}.pdb"
    gt_pdb   = GT_DIR / f"{stem}.pdb"

    ENTRIES.append((fasta_path, pred_pdb, gt_pdb))


if __name__ == "__main__":
    for fasta_file, pred_pdb, gt_pdb in ENTRIES:

        fasta = Path(fasta_file).read_text().splitlines()
        fasta_seq = "".join([x.strip() for x in fasta if not x.startswith(">")])

        result = compare_prediction_to_pdb(fasta_seq, pred_pdb, gt_pdb)

        print("\nGround truth comparison")
        print(f"FASTA: {fasta_file}")
        print(f"PRED : {pred_pdb}")
        print(f"GT   : {gt_pdb}")

        if not result["ok"]:
            print("FAILED:", result["reason"])
        else:
            print(f"Chain: {result['gt_chain']}")
            print(f"Seq identity: {result['seq_identity_pct']:.1f}%")
            print(f"Aligned positions: {result['n_aligned']}")
            print(f"CA-RMSD: {result['CA_RMSD_A']:.2f} Å")
            print(f"TM-score: {result['TM_score']:.2f}")
