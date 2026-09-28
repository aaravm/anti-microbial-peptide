import subprocess
import time
from pathlib import Path

workdir = Path("generate_broad_spectrum/_work")
fasta = workdir / "apex_candidates.fasta"
out = workdir / "apex_pred.csv"
seqs = ">seq1\nACDEFGHIKLMNPQRSTVWY\n" * 3000
fasta.write_text(seqs)
start = time.time()
subprocess.run(
    ["uv", "run", "python", "APEX_predict.py", "-i", str(fasta.resolve()), "-o", str(out.resolve())],
    cwd="apex",
    check=True,
)
print(f"Time for 3000: {time.time() - start:.2f}s")
