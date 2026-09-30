import pandas as pd
from pathlib import Path
import subprocess

def get_mics(fasta_path: Path):
    apex_dir = Path("apex")
    out = Path("temp_apex_out.csv")
    subprocess.run(
        ["uv", "run", "python", "APEX_predict.py", "-i", str(fasta_path.resolve()), "-o", str(out.resolve())],
        cwd=str(apex_dir),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    df = pd.read_csv(out, index_col=0)
    panel = [
        "A. baumannii ATCC 19606", "E. coli ATCC 11775", "E. coli AIC221", "E. coli AIC222",
        "K. pneumoniae ATCC 13883", "P. aeruginosa PA01", "P. aeruginosa PA14",
        "S. aureus ATCC 12600", "S. aureus (ATCC BAA-1556) - MRSA",
        "vancomycin-resistant E. faecalis ATCC 700802", "vancomycin-resistant E. faecium ATCC 700221"
    ]
    
    mean_mics = df[panel].mean(axis=1)
    max_mics = df[panel].max(axis=1)
    
    return mean_mics.mean(), max_mics.mean()

print("\n--- Evaluating Round 1 (Mean MIC Optimized) ---")
r1_mean, r1_max = get_mics(Path("generate_broad_spectrum/top_round1.fasta"))
print(f"Average Mean MIC:       {r1_mean:.1f} µM")
print(f"Average Worst-Case MIC: {r1_max:.1f} µM")

print("\n--- Evaluating Round 2 (Worst-Case MIC Optimized) ---")
r2_mean, r2_max = get_mics(Path("generate_broad_spectrum/top.fasta"))
print(f"Average Mean MIC:       {r2_mean:.1f} µM")
print(f"Average Worst-Case MIC: {r2_max:.1f} µM")
print("\n")
