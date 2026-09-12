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
  finetune.py         EDCLFinetuneModel: pretrained encoder + task head (regression or binary_classification)
  metrics.py          masked MSE/BCE losses (NaN-safe multi-label) + MAE/RMSE/ROC-AUC metrics
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

## Downstream task types

The paper evaluates on both regression benchmarks (QM9, ESOL, FreeSolv,
Lipophilicity, ...) and binary/multi-label classification benchmarks
(BACE, BBBP, ClinTox, HIV, MUV, PCBA, SIDER, Tox21, ToxCast, ...). Set
`model.task_type` in `configs/finetune.yaml` accordingly:

| `task_type` | Loss | Reported metrics | Missing labels |
|---|---|---|---|
| `regression` (default) | `metrics.masked_mse_loss` | MAE, RMSE (`metrics.regression_metrics`) | NaN entries in `y` are masked out of the loss |
| `binary_classification` | `metrics.masked_bce_loss` (BCE-with-logits) | mean ROC-AUC across label columns with both classes present (`metrics.classification_metrics`); columns with only one class, or fully missing, are skipped rather than crashing | NaN entries in `y` (MoleculeNet's per-task missing values) are masked out of both the loss and AUC computation |

`EDCLFinetuneModel`'s task head always outputs raw logits/values (no
final sigmoid) so the same head works for either loss.


## Checkpoints produced

- `train_pretrain.py` saves `{ckpt_dir}/last.pt` (and periodic epoch
  checkpoints) containing `encoder_state_dict` + `encoder_config`, plus
  `ema_encoder_state_dict` (EMA-averaged encoder weights, decay set by
  `train.ema_decay`; omitted if `ema_decay <= 0`).
- `train_finetune.py` loads that checkpoint via
  `EDCLFinetuneModel.from_pretrained(...)` (using the EMA weights by
  default — set `use_ema: false` in `configs/finetune.yaml` to use the raw
  weights instead) and saves the best fine-tuned model (encoder + task
  head) to `{ckpt_dir}/best.pt`.

## Deviations from the paper

This is a faithful-but-scaled-down reimplementation. Known deviations,
tracked here so results are not silently misattributed to the method:

| # | Paper (see `paper.pdf`) | This repo | Why / impact |
|---|---|---|---|
| D1 | Encoder F_phi = Equiformer V2 (SO(3)/e3nn, spherical harmonics up to L_max, Table 9) | `backbone.EquivariantEncoder`: an E(3)-equivariant EGNN (invariant scalar messages + gated equivariant vector channel) | Avoids the heavy `e3nn`/Equiformer-V2 dependency. Same equivariance guarantee (rotation/translation/reflection), smaller model. See `backbone.py` module docstring. |
| D2 | `num_layers` = 20 Equiformer-V2 blocks (Table 9) | `configs/*.yaml`: `num_layers: 4` | CPU-friendly default for the smoke tests in this repo; raise in your own config for full-scale runs. |
| D3 | `max_neighbors` = 50 (Table 9) | `configs/*.yaml`: `max_neighbors: 32` | Same reason as D2. |
| D4 | Training recipe (Table 2): linear LR warmup + cosine annealing, EMA of encoder weights (decay 0.999, used for eval/downstream), dropout 0.2, stochastic depth 0.05/0.1 | Implemented: `schedule.build_warmup_cosine_scheduler` (warmup+cosine, wired into both training scripts via `train.warmup_epochs`/`train.min_lr_ratio`), `ema.EMA` (wired into `train_pretrain.py` via `train.ema_decay`, consumed by `EDCLFinetuneModel.from_pretrained(use_ema=...)`), `backbone.EGNNLayer` dropout + `_drop_path` stochastic depth (wired via `EDCLConfig.dropout`/`drop_path_max`, linearly scheduled across layers) | Fixed in this round (previously flat AdamW with no LR schedule/EMA/regularisation — see `tests/test_training_recipe.py` for behavioural pins). Defaults in `configs/pretrain.yaml`: `dropout=0.2`, `drop_path_max=0.1`, `ema_decay=0.999`, `warmup_epochs=5`, matching Table 2 where the paper gives exact numbers. |
| D5 | Dataset scale / exact benchmarks (e.g. QM9/GEOM subsets, Table 3-8 numbers) | `data.SyntheticMoleculeDataset` fallback when `data.path` is null; real data must be supplied as a validated `.pt` (`docs/data.md`) | This repo ships no dataset; reproducing headline numbers requires the paper's actual training data, which is not redistributed here. |

None of D1-D5 affect the correctness of the loss/objective math (Eq. 1-17
and `docs/paper_equations.md`), which is verified in
`tests/test_losses.py`, `tests/test_validity.py`, `tests/test_equivariance.py`.
