import subprocess
from pathlib import Path

workdir = Path("generate_broad_spectrum/_work")
fasta = workdir / "apex_candidates.fasta"
out = workdir / "apex_pred.csv"
fasta.write_text(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
subprocess.run(
    ["uv", "run", "python", "APEX_predict.py", "-i", str(fasta.resolve()), "-o", str(out.resolve())],
    cwd="apex",
    check=True,
)
print("SUCCESS!")
