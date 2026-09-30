from pathlib import Path
import random

lib_path = Path("generate_broad_spectrum/library.fasta")
mutants_path = Path("generate_broad_spectrum/mutants.fasta")

# Read current library
seqs = set()
with open(lib_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line.startswith(">") and line:
            seqs.add(line)

print(f"Current library sequences: {len(seqs)}")
needed = 50000 - len(seqs)

if needed > 0:
    print(f"Padding with {needed} valid sequences from mutants.fasta...")
    
    # Read mutants
    mutants = set()
    with open(mutants_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">") and line:
                mutants.add(line)
                
    # Find mutants not in library deterministically
    sorted_mutants = sorted(list(mutants))
    available = [m for m in sorted_mutants if m not in seqs]
    random.seed(42)
    random.shuffle(available)
    
    extra = available[:needed]
    
    # Append to library
    with open(lib_path, "a") as f:
        for i, seq in enumerate(extra):
            idx = len(seqs) + i
            f.write(f">seq_{idx}\n{seq}\n")
            
    print(f"Successfully appended {len(extra)} sequences.")
    print("library.fasta is now exactly 50,000 sequences!")
else:
    print("Already 50,000 or more sequences.")
