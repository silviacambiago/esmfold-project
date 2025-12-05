# Instructions to run ESMFold on HPC Cluster Curta

## Prerequisites

- HPC account with GPU access
- Python 3.9 available as a module
- Cloned project repository:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/silviacambiago/esmfold-project.git
cd esmfold-project
```

## Step 1 — Start GPU Session

```bash
salloc \
  --partition=gpu \
  --qos=standard \
  --account=agbelik \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=08:00:00
```

A GPU node (g005, g006, ...) should be available now.

## Step 2 - Load System Modules

```bash
module purge
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
```

Don't install PyTorch via pip, use this module.

## Step 3 — Create Virtual Environment

```bash
python -m venv venv_esmf
source venv_esmf/bin/activate
```

Check that system Python is not used:

```bash
which python
python -c "import sys; print(sys.prefix)"
```

The expected output should be:

```bash
/home/<user>/projects/esmfold-project/venv_esmf
```

## Step 4 - Install Required Python Packages

```bash
pip install --upgrade pip
```
```bash
pip install \
  omegaconf==2.3.0 \
  einops \
  ml-collections \
  biopython==1.85 \
  dm-tree==0.1.8 \
  biotite
```

And also install ESM from this link and not from pip:

```bash
pip install "fair-esm @ git+https://github.com/facebookresearch/esm.git@main"
```

and OpenFold:

```bash
pip install openfold==1.0.0
```

## Step 5 - Patch OpenFold Utils:

Find OpenFold location:

```bash
python - << 'EOF'
import openfold, os
print(os.path.dirname(openfold.__file__))
EOF
```

Go to utils/ subfolder on that path and replace __init__.py:

```bash
cd <printed-path>/utils
cat > __init__.py << 'EOF'

from . import kernel         
from . import seed           
from . import checkpointing 

__all__ = ["kernel", "seed", "checkpointing"]
EOF
```

Now return to project: 

```bash
cd ~/projects/esmfold-project
```

## Step 6 — Verify Installation

```bash
python - << 'EOF'
import torch, esm, openfold
import torch.cuda
print("torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("esm:", esm.__file__)
print("openfold:", openfold.__file__)
EOF
```

If all paths resolve correctly, the environment is valid.

## Step 7 — Run ESMFold

Run the basics example script:

```bash
python fold_v1.py
```

Expected output:

```bash
Using device: cuda
Mean pLDDT: XX.XX
```

If that works, everything else should work, too. 

