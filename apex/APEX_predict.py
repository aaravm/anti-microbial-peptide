"""APEX-pathogen inference — predicts MIC (uM) against the 11-pathogen panel.

Vendored from gitlab machine-biology-group-public/apex-pathogen (the APEX release used to
score the AMP-Diffusion library in Torres et al., Cell Biomaterials 2025). Wrapped for the
starter kit: takes -i FASTA / -o CSV, auto-detects GPU, loads models relative to this file.
"""

import glob
import math
import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from APEX_models import AMP_model  # noqa: F401 (unpickled models reference this class)
from utils import make_vocab, onehot_encoding

parser = OptionParser()
parser.add_option("-i", "--i", default="./test_seqs.fasta", help="input FASTA path")
parser.add_option("-o", "--o", default="Predicted_MICs.csv", help="output CSV path")
parser.add_option("-g", "--g", default="auto", help="gpumode: 1=GPU, 0=CPU, auto=detect")
(opts, args) = parser.parse_args()

data_path = str(opts.i)
out_path = str(opts.o)
useGPU = ("1" if torch.cuda.is_available() else "0") if str(opts.g) == "auto" else str(opts.g)

pathogen_list = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]

max_len = 52  # start char + 50 aa + end char; longer peptides are truncated
word2idx, _ = make_vocab()

# Load the 8 pretrained APEX-pathogen models (relative to this file, not the cwd).
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "APEX_pathogen_models")
APEX_models = []
for a_model in sorted(glob.glob(os.path.join(MODEL_DIR, "APEX_*"))):
    model = torch.load(a_model, map_location="cpu")
    model.eval()
    APEX_models.append(model)

# Load input peptides.
seq_list = []
for fasta in SeqIO.parse(open(data_path), "fasta"):
    sequence = str(fasta.seq)
    if len(sequence) <= 50:
        seq_list.append(sequence)
if len(seq_list) == 0:
    print("No sequences were loaded")
    sys.exit(1)
seq_list = np.array(seq_list)

batch_size = 256  # changed from 3000 to avoid OOM on CPU

# Predict per-species MIC (uM); average the 8 base learners and compute variance.
all_preds = []
all_preds_raw = []

for ensemble_id in range(len(APEX_models)):
    if useGPU == "1":
        AMP_model = APEX_models[ensemble_id].cuda().eval()
    else:
        AMP_model = APEX_models[ensemble_id].cpu().eval()

    data_len = len(seq_list)
    for i in range(int(math.ceil(data_len / float(batch_size)))):
        seq_batch = seq_list[i * batch_size:(i + 1) * batch_size]
        seq_rep = onehot_encoding(seq_batch, max_len, word2idx)
        if useGPU == "1":
            X_seq = torch.LongTensor(seq_rep).cuda()
            AMP_pred_batch_raw = AMP_model(X_seq).cpu().detach().numpy()
        else:
            X_seq = torch.LongTensor(seq_rep)
            AMP_pred_batch_raw = AMP_model(X_seq).detach().numpy()

        AMP_pred_raw = AMP_pred_batch_raw if i == 0 else np.vstack([AMP_pred_raw, AMP_pred_batch_raw])
        
        # Training target was -log10(MIC / 1e6); invert to MIC in uM.
        AMP_pred_batch = 10 ** (6 - AMP_pred_batch_raw)
        AMP_pred = AMP_pred_batch if i == 0 else np.vstack([AMP_pred, AMP_pred_batch])

    all_preds.append(AMP_pred)
    all_preds_raw.append(AMP_pred_raw)

# shape: (8, N_sequences, 11_pathogens)
all_preds = np.stack(all_preds, axis=0)
all_preds_raw = np.stack(all_preds_raw, axis=0)

AMP_mean = np.mean(all_preds, axis=0)
# Compute variance on the RAW scale so confidence isn't penalized by the exponential
AMP_var = np.var(all_preds_raw, axis=0)

# Combine into one DataFrame: first 11 cols are means, next 11 cols are variances
var_columns = [p + "_var" for p in pathogen_list]
combined_data = np.hstack([AMP_mean, AMP_var])
df = pd.DataFrame(data=combined_data, columns=pathogen_list + var_columns, index=seq_list)
df.to_csv(out_path)
