import argparse
from pathlib import Path

def read_fasta(path: Path) -> list[str]:
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seqs.append(line)
    return seqs

def write_fasta(seqs: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(seqs, start=1):
            f.write(f">mutant_{i}\n{seq}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate 1-edit mutants for local search.")
    parser.add_argument("--input", type=Path, default=Path("generate_broad_spectrum/top.fasta"))
    parser.add_argument("--output", type=Path, default=Path("generate_broad_spectrum/mutants.fasta"))
    args = parser.parse_args()

    STANDARD_AA_SET = "ACDEFGHIKLMNPQRSTVWY"

    print(f"Loading {args.input}...")
    originals = read_fasta(args.input)
    print(f"Loaded {len(originals)} sequences.")

    mutants = set(originals)  # Keep the originals in the pool!

    print("Generating all single-residue substitutions...")
    for seq in originals:
        for i in range(len(seq)):
            for aa in STANDARD_AA_SET:
                if aa != seq[i]:
                    mutant = seq[:i] + aa + seq[i+1:]
                    mutants.add(mutant)

    mutants_list = list(mutants)
    print(f"Total unique sequences generated: {len(mutants_list)}")

    write_fasta(mutants_list, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
