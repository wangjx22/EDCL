# EDCL — Equivariant Denoising Contrastive Learning



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


## Run tests

```bash
python -m pytest -q
```


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
  data.py             validated .pt loading, MoleculeBatch, collation, synthetic fixture
tests/                pytest suite (unit + integration; run `pytest -q` for current count)
scripts/              train_pretrain.py, train_finetune.py (CLI)
configs/              pretrain.yaml, finetune.yaml
docs/                 data contract and paper equation traceability
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
| Energy auxiliary loss (supervised MSE vs. label, clean branch only; labels required when beta is non-zero) | `losses.energy_mse_loss` |
| Combined objective (Eq. 17) | `losses.total_edcl_loss`, `losses.EDCLLossWeights` (alpha=0.1, beta=10, lambda=1) |
| Fine-tuning protocol | `finetune.EDCLFinetuneModel` |


## Checkpoints produced

- `train_pretrain.py` saves `{ckpt_dir}/last.pt` (and periodic epoch
  checkpoints) containing `encoder_state_dict` + `encoder_config`.
- `train_finetune.py` loads that checkpoint via
  `EDCLFinetuneModel.from_pretrained(...)` and saves the best fine-tuned
  model (encoder + task head) to `{ckpt_dir}/best.pt`.
