# EDCL — Equivariant Denoising Contrastive Learning (Rewrite)

A from-scratch, dependency-light reimplementation of **EDCL**, following the
methodology in the accompanying paper (`EDCL_work/paper.txt`). This rewrite
replaces the original messy repository (`EDCL_work/orig_repo/`) with a small,
fully-tested `src/edcl` package plus CLI training scripts.

> Original repo: https://github.com/wangjx22/EDCL

## Why a rewrite?

The original repository mixed experiment scaffolding, hard-coded paths, and
partial implementations. This rewrite:
- Implements every paper equation as a **standalone, unit-tested function**
  (`src/edcl/losses.py`, `validity.py`, `noise_generator.py`), each docstring
  citing the corresponding equation number.
- Provides a genuinely **SE(3)/E(3)-equivariant encoder** (see Deviations
  below) with numeric equivariance tests, not just "should be invariant"
  comments.
- Ships **runnable CLI trainers** (`scripts/train_pretrain.py`,
  `scripts/train_finetune.py`) that work out of the box on a synthetic
  dataset (no external data required), and can be pointed at real molecular
  data via a `.pt` file.
- Has **zero hard dependency on `torch_geometric` / `e3nn`**: a minimal
  scatter/radius-graph implementation lives in `src/edcl/ops.py`, so the
  whole package installs with just `torch` + `pyyaml` (see
  `requirements.txt`).

## Install

```bash
pip install -r requirements.txt
pip install -e .   # optional; or just add src/ to PYTHONPATH
```

## Quick start

```bash
# 1. Pretrain (dual-branch denoising + contrastive + energy objective)
python scripts/train_pretrain.py --config configs/pretrain.yaml

# 2. Fine-tune the pretrained encoder on a downstream target
python scripts/train_finetune.py --config configs/finetune.yaml
```

Both configs default to a synthetic in-memory dataset (`data.path: null`) so
the full pipeline is runnable immediately for smoke-testing. To use real
data, point `data.path` at a `.pt` file containing `list[dict(z, pos[, y])]`
(see `src/edcl/data.py` docstring) and set the path in the YAML config.

## Run tests

```bash
python -m pytest -q
```

16 tests covering: numeric SE(3)/E(3) equivariance of the encoder and
denoising head, all four loss terms (denoise NLL, KL prior, InfoNCE
contrastive, energy MSE) against hand-derived reference values, the r_min
validity-constrained perturbation sampler, and an end-to-end forward+backward
smoke test of the full dual-branch model.

## Repository layout

```
src/edcl/
  ops.py              minimal scatter_sum/scatter_mean/radius_graph (no torch_geometric)
  backbone.py         EquivariantEncoder (EGNN-style, shared F_phi for both branches)
  noise_generator.py  G_theta: per-atom adaptive noise scale sigma_i (Eq. ~8-9)
  validity.py         r_min-constrained rejection/resample perturbation sampler
  heads.py            DenoiseHead (equivariant), EnergyHead, ProjectionHead, TaskHead
  losses.py           denoising NLL, KL prior, InfoNCE contrastive, energy MSE, EDCLLossWeights
  model.py            EDCLPretrainModel: dual-branch forward + combined loss (Eq. 17)
  finetune.py         EDCLFinetuneModel: pretrained encoder + task head
  data.py             MoleculeBatch, collate_molecules, SyntheticMoleculeDataset
tests/                pytest suite (16 tests, all passing)
scripts/              train_pretrain.py, train_finetune.py (CLI)
configs/              pretrain.yaml, finetune.yaml
docs/paper_equations.md   equation -> code traceability table
```

## Equation → code map

See `docs/paper_equations.md` for the full table. Summary:

| Paper concept | Code |
|---|---|
| Encoder F_phi | `backbone.EquivariantEncoder` |
| Noise generator G_theta, sigma_i | `noise_generator.NoiseGenerator` |
| Perturbation r_tilde = r + sigma*eps (validity-constrained) | `validity.sample_valid_perturbation` |
| Denoising NLL loss (weighted MSE / sigma_i^2, no log term) | `losses.denoising_nll_loss` |
| KL(sigma prior) regularizer (closed-form Gaussian KL) | `losses.kl_prior_loss` |
| Contrastive InfoNCE loss (asymmetric, noisy->clean) | `losses.info_nce_contrastive_loss` |
| Energy auxiliary loss (supervised MSE vs. label, clean branch only; 0 if no label) | `losses.energy_mse_loss` |
| Combined objective (Eq. 17) | `losses.total_edcl_loss`, `losses.EDCLLossWeights` (alpha=0.1, beta=10, lambda=1) |
| Fine-tuning protocol | `finetune.EDCLFinetuneModel` |

## Known deviations from the paper (and why)

1. **Encoder architecture**: the paper uses a full **Equiformer V2**
   (spherical-harmonic features up to degree `L_max`, Wigner-D equivariant
   tensor products; see paper Table 9). That requires `e3nn`, which is not
   available/installable in this environment within the time budget. We
   substitute a **genuinely SE(3)/E(3)-equivariant EGNN** (Satorras et al.,
   ICML 2021) that satisfies the same invariance/equivariance contract
   (verified numerically in `tests/test_equivariance.py`, rotation error
   ~1e-7). This is strictly less expressive (no higher-order tensor
   features) but preserves every symmetry property the loss functions rely
   on.
2. **No `torch_geometric` dependency**: `scatter_sum`/`scatter_mean`/
   `radius_graph` are reimplemented from scratch in `src/edcl/ops.py`
   (dense O(N^2) radius graph — fine for small molecules / unit tests, not
   optimized for large-scale batched training).
3. **Loss weight correction**: the original repo's loss aggregation had the
   contrastive/energy coefficients swapped relative to the paper's stated
   alpha=0.1 (contrastive), beta=10 (energy), lambda=1 (KL). This rewrite
   uses the paper's stated values as documented defaults in
   `EDCLLossWeights`.
4. **Added validity check**: the paper specifies that perturbed positions
   must respect a minimum inter-atomic distance `r_min`; the original repo
   sampled Gaussian noise unconditionally. This rewrite adds
   `validity.sample_valid_perturbation`, a bounded rejection/resample loop
   (tested in `tests/test_validity.py`).
5. **Synthetic dataset fallback**: real pretraining datasets (PCQM4Mv2/OC20/
   QM9-scale) are not available in this sandboxed environment; the CLI
   scripts fall back to a synthetic random-molecule dataset so the full
   pipeline is exercised end-to-end (forward, backward, checkpointing).
   Swap in real data via `data.path` pointing at a prepared `.pt` file.

## Checkpoints produced

- `train_pretrain.py` saves `{ckpt_dir}/last.pt` (and periodic epoch
  checkpoints) containing `encoder_state_dict` + `encoder_config`.
- `train_finetune.py` loads that checkpoint via
  `EDCLFinetuneModel.from_pretrained(...)` and saves the best fine-tuned
  model (encoder + task head) to `{ckpt_dir}/best.pt`.
