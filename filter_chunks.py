import sys
from pathlib import Path

# Add the 'src' directory so Python can find the starter kit's code
sys.path.insert(0, str(Path("src").resolve()))

from ampdiffusion_starter_kit.generate import (
    _read_fasta_sequences,
    _write_fasta,
    select_top,
    ANTIBACTERIAL_FASTA,
    TRAINING_FASTA,
)

def main():
    chunks_dir = Path("amp-diffusion-chunks")
    
    print(f"Reading generated sequences from {chunks_dir}...")
    all_sequences = set()
    
    # Read all chunk files and deduplicate sequences
    for fasta_file in chunks_dir.glob("*.fasta"):
        print(f"  Reading {fasta_file.name}...")
        seqs = _read_fasta_sequences(fasta_file)
        all_sequences.update(seqs)
    
    unique_library = list(all_sequences)
    print(f"Total unique sequences after merging: {len(unique_library)}")

    out_dir = Path("generate_broad_spectrum")
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_work"

    # Save the merged library for your records
    library_path = out_dir / "library.fasta"
    _write_fasta(unique_library, library_path)
    print(f"Wrote merged library -> {library_path}")

    # Load references
    print("Loading reference and training sequences...")
    references = set(_read_fasta_sequences(Path(ANTIBACTERIAL_FASTA)))
    known_amps = _read_fasta_sequences(Path(TRAINING_FASTA))

    # Run the filtering step
    print("Running APEX scorer and novelty/diversity filters (this may take a bit on CPU)...")
    top_100 = select_top(unique_library, top_k=100, references=references, known_amps=known_amps, workdir=workdir)
    
    top_path = out_dir / "top.fasta"
    _write_fasta(top_100, top_path)
    print(f"\nSuccess! Wrote top {len(top_100)} sequences to {top_path}")

if __name__ == "__main__":
    main()
