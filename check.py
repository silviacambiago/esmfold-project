from esm import pretrained
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = pretrained.esmfold_v1().to(device)
print("model first param device:", next(model.parameters()).device)
