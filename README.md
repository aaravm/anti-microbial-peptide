# AMP-Diffusion Starter Kit: AMP Challenge 2027 Baseline

A self-contained, reproducible generator that produces the **AMP-Diffusion baseline library**
for the [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027).

AMP-Diffusion (Torres et al., [*Cell Biomaterials* 2025](https://doi.org/10.1016/j.celbio.2025.100183);
Chen et al., [bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2024.03.03.583201v1)) is a
latent-space diffusion model that de novo generates antimicrobial peptides in the ESM-2
embedding space. It is an official challenge **baseline** (excluded from rankings), packaged
here behind the `generate_*` entry-point contract so it runs with a single `uv run` and emits
a validator-compliant library.

## Abstract

AMP-Diffusion denoises Gaussian noise into ESM-2 embeddings and decodes them to sequences,
generating peptides that match experimentally validated AMPs in pseudo-perplexity, amino-acid
diversity, and physicochemical properties. This baseline uses AMP-Diffusion's default
unconstrained generation with deterministic decoding, applying only automated, deterministic
filters: the challenge's hard sequence rules for the full library, plus the paper's
computational filtering (APEX activity, novelty, and diversity) for the top-100. No sequence
optimization, prompt engineering, or manual curation is performed.

## What this produces

Running the entry point writes two files under `generate_broad_spectrum/`:

```
generate_broad_spectrum/
  library.fasta   full 50,000-sequence library
  top.fasta       top-100 ranked candidates
```

AMP-Diffusion is strain-agnostic, so only the **broad-spectrum** category is wired up.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and a **CUDA GPU** (the diffusion sampler and the
APEX ensemble are GPU workloads; see [Compute](#compute-requirements)). The pinned model
checkpoint is committed under `checkpoint/` via Git LFS, so no data download is needed.

```bash
git lfs install && git lfs pull   # fetch checkpoint/model.pt and the APEX weights
uv run generate_broad_spectrum
```

Optional arguments (all have defaults; the defaults reproduce the baseline):

| Flag | Default | Description |
|------|---------|-------------|
| `--n-sequences` | `50000` | Number of sequences in the library |
| `--top-k` | `100` | Number of top-ranked sequences |
| `--seed` | `42` | Random seed (fixed for reproducibility) |
| `--batch-size` | `256` | Diffusion sampling batch size |
| `--checkpoint` | `checkpoint/model.pt` | AMP-Diffusion EMA checkpoint (16.54M params) |
| `--antibacterial-fasta` | `data/antibacterial.fasta` | Challenge reference set for the overlap / novelty guard |
| `--known-amps-fasta` | `data/training/training.fasta` | Known-AMP set for the paper's novelty filter |
| `--device` | `cuda` if available else `cpu` | Compute device |

Output is reproducible across repeated runs on the same machine (fixed seeds +
`torch.use_deterministic_algorithms`).

## How it works

`generate_broad_spectrum` samples Gaussian noise, denoises it into ESM2 embeddings with the
AMP-Diffusion model, and decodes the embeddings to sequences via the ESM2 language-model head
(argmax over the 20 standard amino acids). This is AMP-Diffusion's **default** de-novo
generation; no conditioning or optimization is applied.

**Library filters (applied to the 50,000-sequence library):**

1. *Alphabet*: only the 20 standard amino acids (`ACDEFGHIKLMNPQRSTVWY`).
2. *Length*: 8-50 residues (lengths sampled in 10-40).
3. *Uniqueness*: global deduplication across the whole library.
4. *No known-antibacterial overlap*: drop any sequence identical to one in
   `data/antibacterial.fasta`.

**Top-100 selection — the paper's computational filtering** (Torres et al. 2025), which
selects for activity, novelty, *and* diverse sequence-space coverage:

1. *Activity*: every library sequence is scored by the **APEX** ensemble (8 models) for mean
   predicted MIC across the 11-pathogen panel; candidates are taken most-potent-first,
   preferring those with **MIC ≤ 64 µM**.
2. *Novelty*: drop any peptide with **> 0.60 local-alignment similarity** to a known AMP (the
   DRAMP/APD3/DBAASP training set in `data/training/`).
3. *Diversity*: among the remaining peptides, of any pair more than **0.40 local-alignment
   similar**, keep only the one with the lower predicted MIC. The 100 most-potent peptides
   passing all three criteria form the top-100; if the ≤ 64 µM pool yields fewer than 100, the
   remaining slots are filled by the next-most-potent peptides beyond the cut.

A final guard enforces the challenge's own **< 80% Levenshtein** rule vs `data/antibacterial.fasta`.
The similarity metric is a documented Smith-Waterman operationalization (see
[`scoring`/`similarity`](src/ampdiffusion_starter_kit/similarity.py)); the paper's further
wet-lab preselection steps are, by construction, not reproduced here.

### APEX runs in its own isolated environment

Candidates are scored with **APEX-pathogen** ([`machine-biology-group-public/apex-pathogen`](https://gitlab.com/machine-biology-group-public/apex-pathogen)),
the release covering the 11 clinical pathogens. It is vendored under [`apex/`](apex/) as a
**separate `uv` project** invoked as a subprocess, so its dependencies never co-resolve with the
kit's ESM2 stack. The 8 ensemble weights are committed under `apex/APEX_pathogen_models/` via
Git LFS, and the environment syncs on the first APEX call.

## Compute requirements

Full generation is a **GPU workload**: the 1000-step sampler runs ~50,000 times to build the
library. The 8-model APEX scoring step runs on CPU. A full CPU run of the sampler is
impractically slow — use CPU only for small `--n-sequences` smoke tests.

## Model & data provenance

- **Inference code:** the AMP-Diffusion model architecture (`Denoise_Transformer` +
  `GaussianDiffusion1D`) in [`src/ampdiffusion_starter_kit/model.py`](src/ampdiffusion_starter_kit/model.py),
  from the published [`programmablebio/amp-diffusion`](https://github.com/programmablebio/amp-diffusion) repo.
- **Weights:** the released 16.54M-parameter checkpoint (7.40M ESM2-8M attention layers +
  9.14M own parameters), committed under `checkpoint/model.pt` (Git LFS).
- **Decoder / embeddings:** ESM2-8M (`esm2_t6_8M_UR50D`), downloaded by `fair-esm` on first run.
- **APEX scorer:** [`machine-biology-group-public/apex-pathogen`](https://gitlab.com/machine-biology-group-public/apex-pathogen)
  (APEX, [*Nat. Microbiol.* 2025](https://www.nature.com/articles/s41564-025-02061-0)),
  vendored under `apex/` with its 8 ensemble weights — the release the paper used for scoring.
- **Training data:** DRAMP 3.0 + APD3 + DBAASP (public); see [`data/README.md`](data/README.md).
- **`data/antibacterial.fasta`:** the challenge reference set, used only for the overlap and
  novelty checks.
- **Experimental data:** [`experimental/`](experimental/) holds AMP-Diffusion's wet-lab MIC
  measurements; see [`experimental/README.md`](experimental/README.md).

## Project Structure

Follows the [challenge template](https://github.com/szczurek-lab/amp-challenge-2027):

```
ampdiffusion-starter-kit/
├── checkpoint/model.pt   # AMP-Diffusion checkpoint (Git LFS)
├── apex/                 # vendored APEX scorer (isolated uv project, LFS weights)
├── data/                 # antibacterial reference set + training-data disclosure
├── experimental/         # synthesized peptides + measured MICs
├── metrics/              # Phase-1 seqme metrics (added at launch)
├── scripts/              # verify_submission.py (the challenge validator)
└── src/ampdiffusion_starter_kit/
    ├── generate.py       # generate_broad_spectrum entry point
    ├── scoring.py        # APEX MIC scorer
    ├── similarity.py     # local-alignment similarity (novelty + diversity filters)
    └── model.py          # AMP-Diffusion model architecture
```

## Work in progress

- **`metrics/`**: Phase-1 `seqme` metrics ship at competition launch.

## License

MIT (see [LICENSE](LICENSE)). AMP-Diffusion and APEX-pathogen are released by their respective
authors; see their repositories for terms.
