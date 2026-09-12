# Paper Equations → Code Traceability

This table maps concepts from the EDCL paper to the exact function or class
implementing them.

## 1. Encoder

| Concept | Code |
|---|---|
| Shared equivariant encoder F_phi(z, r) -> (h_i, v_i, h_graph) | `backbone.EquivariantEncoder.forward` |
| Per-atom invariant scalar features h_i | `EncoderOutput.h` |
| Per-atom equivariant vector features v_i (transform with rotations) | `EncoderOutput.v` |
| Graph-level invariant readout h_graph = mean_i h_i | `EncoderOutput.h_graph` (scatter_mean) |

## 2. Perturbation / noise branch

| Concept | Code |
|---|---|
| Adaptive per-atom noise scale sigma_i = G_theta(h_i) | `noise_generator.NoiseGenerator.forward` |
| softplus(.) + eps ensures sigma_i > 0 | same, `nn.Softplus()` + `self.eps` |
| Perturbed coordinates r_tilde_i = r_i + sigma_i * eps_i, eps_i ~ N(0, I) | `validity.sample_valid_perturbation` |
| Validity constraint: reject samples with pairwise distance < r_min, resample | `validity.sample_valid_perturbation` (bounded retry loop) |

## 3. Losses

| Paper loss | Formula (as implemented) | Code |
|---|---|---|
| Denoising NLL | `L_denoise = mean_i( \|\|Delta_hat_i - Delta_i\|\|^2 / sigma_i^2 )`, `Delta_i = r_tilde_i - r_i` (Eq. 10/12) | `losses.denoising_nll_loss` |
| KL prior on sigma | `L_KL = mean_i[ log(omega/sigma_i) + sigma_i^2/(2*omega^2) - 0.5 ]` = `KL(N(0,sigma_i^2) \|\| N(0,omega^2))` closed form (Eq. 14) | `losses.kl_prior_loss` |
| Contrastive (InfoNCE) | Asymmetric InfoNCE: each noisy-branch graph embedding matched to its own clean-branch embedding among all clean embeddings in the batch (cosine sim / tau, cross-entropy) (Eq. 4) | `losses.info_nce_contrastive_loss` |
| Energy auxiliary | `L_energy = MSE(EnergyHead(h_graph_clean), target_energy)`, supervised against a reference energy label from the data batch; predicted only from the CLEAN branch (Eq. 16). Skipped (0) when the batch carries no energy label. | `losses.energy_mse_loss` |
| Combined objective (Eq. 17) | `L_total = L_denoise + lambda*L_KL + alpha*L_contrast + beta*L_energy`, `alpha=0.1, beta=10, lambda=1` | `losses.total_edcl_loss`, `losses.EDCLLossWeights` |

## 4. Heads

| Concept | Code |
|---|---|
| Equivariant denoising head: `Delta_hat_i = MLP(h_i) ⊙-gate applied to v_i` (linear-in-v to preserve equivariance) | `heads.DenoiseHead` |
| Energy head: invariant scalar from h_graph | `heads.EnergyHead` |
| Projection head for contrastive embedding z | `heads.ProjectionHead` |
| Downstream task head (fine-tuning) | `heads.TaskHead` |

## 5. Training procedures

| Concept | Code |
|---|---|
| Dual-branch pretraining forward (clean + noisy branch share F_phi) | `model.EDCLPretrainModel.forward` |
| Pretraining loop / checkpointing (CLI) | `scripts/train_pretrain.py` |
| Fine-tuning: load encoder weights, attach `TaskHead`, train on labeled targets | `finetune.EDCLFinetuneModel.from_pretrained`, `scripts/train_finetune.py` |

## 6. Supporting infra (not in paper, implementation necessities)

| Need | Code |
|---|---|
| Radius graph / neighbor list without `torch_geometric` | `ops.radius_graph` |
| Segment-scatter sum/mean without `torch_geometric` | `ops.scatter_sum`, `ops.scatter_mean` |
| Batched variable-size-molecule collation | `data.collate_molecules`, `data.MoleculeBatch` |
| Synthetic dataset for dependency-free smoke testing | `data.SyntheticMoleculeDataset` |

## 7. Tests validating the above

| Test file | Validates |
|---|---|
| `tests/test_equivariance.py` | Numeric SE(3)/E(3) equivariance of encoder (h,h_graph invariant; v equivariant) and DenoiseHead output equivariance under random rotations, translations |
| `tests/test_losses.py` | Each loss term against hand-derived reference values (e.g. NLL reduces to Gaussian NLL for constant sigma; KL matches closed-form N(0,s1^2)\|\|N(0,s2^2); InfoNCE matches manual softmax-cross-entropy; energy loss matches MSE) |
| `tests/test_validity.py` | Perturbation sampler never returns configurations violating `r_min`; falls back safely if unsatisfiable within retry budget |
| `tests/test_model_smoke.py` | Full forward+backward pass produces finite loss and non-null gradients for all trainable parameters (except the identity-shortcut branch of ProjectionHead, documented as expected) |
| `tests/test_data_contract.py` | Serialized-data validation, strict collation, and the conditional energy-target contract |
