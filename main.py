import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import QM9


class PretrainConfig:
    def __init__(self):
        # --- Path & Device ---
        self.save_dir = "./checkpoints_pretrain"
        self.data_root = "./data"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Model Architecture ---
        self.hidden_dim = 128
        self.num_layers = 6
        self.num_heads = 4
        
        # --- Physics-Guided Pre-training Params ---
        self.r_min = 0.8           # Min distance for validity check (Angstrom)
        self.omega = 0.1           # Prior std for KL regularization (Eq. 13)
        self.tau = 0.1             # Contrastive temperature (Eq. 6)
        
        # --- Loss Balancing (Eq. 16) ---
        # Alpha (Contrastive), Beta (Energy), Lambda (KL)
        self.alpha = 0.1           
        self.beta = 1.0            # Adjusted: assuming normalized energy targets
        self.lambda_kl = 0.01      # KL term usually needs small weight to not over-regularize
        
        # --- Training Hyperparams ---
        self.batch_size = 64       # Larger batch size benefits Contrastive Learning
        self.lr = 5e-4
        self.weight_decay = 1e-5
        self.epochs = 100
        self.warmup_epochs = 5
        self.grad_clip = 5.0       # Essential for Graph Transformers
        self.ema_decay = 0.999     # Exponential Moving Average decay
        self.mixed_precision = True # Use AMP
        
        # --- Data Specifics (QM9) ---
        # Target 7 is HOMO, 12 is U0 (Energy). We use one as an auxiliary task.
        self.energy_target_idx = 7 


class EMA:
    """
    Exponential Moving Average.
    Crucial for pre-training: The EMA weights often generalize better 
    to downstream tasks than the final raw weights.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def state_dict(self):
        return self.shadow

class LabelNormalizer:
    """
    Normalize energy labels (y - mean) / std.
    Without this, MSE Loss for energy (e.g., -10000 Hartree) 
    will explode and dominate the Denoising Loss.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, loader, target_idx):
        print("Computing dataset statistics for normalization...")
        all_vals = []
        # Iterate a subset or full dataset to compute stats
        for i, batch in enumerate(loader):
            if batch.y is not None:
                val = batch.y[:, target_idx]
                all_vals.append(val)
            if i > 100: break # Approximate with first 100 batches for speed
        
        if len(all_vals) > 0:
            all_vals = torch.cat(all_vals)
            self.mean = all_vals.mean().item()
            self.std = all_vals.std().item()
            print(f"Stats: Mean={self.mean:.4f}, Std={self.std:.4f}")
        else:
            print("Warning: No labels found for normalization.")
            self.mean, self.std = 0.0, 1.0

    def normalize(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-6)


class EquiformerBackbone(nn.Module):
    """
    [Placeholder] Real EquiformerV2 / SE(3)-Transformer should be plugged here.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Linear(1, dim) # Input Z (atomic number)
        # Mock Transformer Layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim*2, 
                                       dropout=0.1, batch_first=False, norm_first=True)
            for _ in range(4)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, data):
        # 1. Embedding
        x = self.embedding(data.x.float()) 
        
        # 2. Equivariant Layers (Mocked)
        # In real code: x = self.equiformer_blocks(x, data.pos, data.batch)
        x = x.unsqueeze(1) 
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x.squeeze(1))
        
        # 3. Graph Pooling (Readout)
        h_graph = global_mean_pool(x, data.batch)
        return x, h_graph

class NoiseGenerator(nn.Module):
    """G_theta: Predicts noise scale sigma per atom."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Softplus() # Ensure sigma > 0
        )

    def forward(self, x):
        # Add epsilon to prevent numerical instability in division later
        return self.net(x) + 1e-6 

class PretrainModel(nn.Module):
    """
    The entire Pre-training Framework.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = EquiformerBackbone(config.hidden_dim)
        self.noise_gen = NoiseGenerator(config.hidden_dim)
        
        # --- Prediction Heads ---
        # 1. Denoising Head (Equivariant -> Shift Vectors)
        self.head_denoise = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3) # Output: [dx, dy, dz]
        )
        
        # 2. Energy Head (Invariant -> Scalar)
        self.head_energy = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # 3. Contrastive Projection Head
        self.head_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, data):
        """
        Forward pass implementing the Dual-Branch logic.
        """
        # === Branch 1: Ground Truth (Clean) ===
        h_gt, g_gt = self.encoder(data)
        
        # Task A: Energy Prediction (Supervised)
        # Predict atom-wise energy, then pool
        atom_energy = self.head_energy(h_gt)
        pred_energy = global_mean_pool(atom_energy, data.batch) # [B, 1]
        
        # Task B: Noise Generation & Sampling
        sigma = self.noise_gen(h_gt) # [N, 1]
        epsilon = torch.randn_like(data.pos)
        pos_tilde = data.pos + sigma * epsilon # Eq. 1
        
        # === Branch 2: Perturbed Structure (Noisy) ===
        data_tilde = data.clone()
        data_tilde.pos = pos_tilde
        # Encode perturbed structure using SHARED encoder
        h_tilde, g_tilde = self.encoder(data_tilde)
        
        # Task C: Denoising (Predict Shifts)
        pred_shifts = self.head_denoise(h_tilde)
        
        # Task D: Contrastive Learning Projections
        z_gt = self.head_proj(g_gt)
        z_tilde = self.head_proj(g_tilde)
        
        return {
            "z_gt": z_gt, "z_tilde": z_tilde,           # Contrastive
            "pred_shifts": pred_shifts,                 # Denoise
            "pred_energy": pred_energy,                 # Energy
            "pos_gt": data.pos, "pos_tilde": pos_tilde, # Coordinates
            "sigma": sigma                              # Noise parameters
        }


class PretrainLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, out, batch_energy_target):
        # --- 1. Contrastive Loss (InfoNCE) ---
        z1 = F.normalize(out['z_gt'], dim=1)
        z2 = F.normalize(out['z_tilde'], dim=1)
        
        logits = torch.mm(z2, z1.t()) / self.cfg.tau
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_con = F.cross_entropy(logits, labels)

        # --- 2. Denoising Loss (Eq. 12 + 13) ---
        target_shift = out['pos_tilde'] - out['pos_gt']
        pred_shift = out['pred_shifts']
        sigma = out['sigma']
        
        # NLL (Weighted MSE)
        # Loss = sum ||s - s_hat||^2 / sigma^2
        mse = torch.sum((pred_shift - target_shift)**2, dim=1, keepdim=True)
        loss_nll = torch.mean(mse / (sigma**2 + 1e-9))
        
        # KL Regularization
        # KL(N(0, sigma^2) || N(0, omega^2))
        var_post = sigma**2
        var_prior = self.cfg.omega**2
        kl = torch.log(self.cfg.omega / (sigma + 1e-9)) + (var_post / (2*var_prior)) - 0.5
        loss_kl = torch.mean(kl)
        
        loss_denoise = loss_nll + self.cfg.lambda_kl * loss_kl

        # --- 3. Energy Loss ---
        # batch_energy_target should be NORMALIZED already
        loss_energy = F.mse_loss(out['pred_energy'], batch_energy_target.view(-1, 1))

        # --- Total Loss ---
        total = loss_denoise + \
                self.cfg.alpha * loss_con + \
                self.cfg.beta * loss_energy
        
        return total, {
            "total": total.item(), 
            "con": loss_con.item(), 
            "nll": loss_nll.item(), 
            "kl": loss_kl.item(), 
            "eng": loss_energy.item()
        }

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        
        # 1. Data Loading
        print(f"Loading Data (QM9)...")
        # Ensure QM9 is downloaded. 
        self.dataset = QM9(root=cfg.data_root)
        
        # 2. Normalization Setup
        self.normalizer = LabelNormalizer()
        tmp_loader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        self.normalizer.fit(tmp_loader, cfg.energy_target_idx)
        
        # 3. Final Loader
        self.loader = DataLoader(self.dataset, batch_size=cfg.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        
        # 4. Model & Optim
        self.model = PretrainModel(cfg).to(cfg.device)
        self.ema = EMA(self.model, cfg.ema_decay)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = self._get_scheduler()
        self.criterion = PretrainLoss(cfg)
        
        # 5. Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda' if cfg.device.type == 'cuda' else 'cpu')

    def _get_scheduler(self):
        # Linear Warmup + Cosine Annealing
        def lr_lambda(epoch):
            if epoch < self.cfg.warmup_epochs:
                return (epoch + 1) / self.cfg.warmup_epochs
            progress = (epoch - self.cfg.warmup_epochs) / (self.cfg.epochs - self.cfg.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def save_checkpoint(self, epoch, metrics):
        # Save structure compatible with Fine-tuning script
        state = {
            'epoch': epoch,
            'config': self.cfg.__dict__,
            'state_dict': self.model.state_dict(),     # Standard weights
            'ema_state_dict': self.ema.state_dict(),   # EMA weights (Preferred for finetuning)
            'optimizer': self.optimizer.state_dict(),
            'normalizer': {'mean': self.normalizer.mean, 'std': self.normalizer.std}, # Save norm stats
            'metrics': metrics
        }
        path = os.path.join(self.cfg.save_dir, f"ckpt_epoch_{epoch}.pt")
        torch.save(state, path)
        print(f"Saved Checkpoint: {path}")

    def train_epoch(self, epoch):
        self.model.train()
        stats = {"total": 0, "con": 0, "nll": 0, "kl": 0, "eng": 0}
        steps = 0
        
        for batch in self.loader:
            batch = batch.to(self.cfg.device)
            
            # Prepare Normalized Targets
            raw_energy = batch.y[:, self.cfg.energy_target_idx]
            norm_energy = self.normalizer.normalize(raw_energy)
            
            self.optimizer.zero_grad()
            
            # AMP Context
            with torch.amp.autocast('cuda' if self.cfg.device.type == 'cuda' else 'cpu', 
                                    enabled=self.cfg.mixed_precision):
                out = self.model(batch)
                loss, loss_items = self.criterion(out, norm_energy)
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Unscale & Clip Grads
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            
            # Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA
            self.ema.update(self.model)
            
            # Logging
            for k, v in loss_items.items(): stats[k] += v
            steps += 1
        
        # Avg stats
        return {k: v/steps for k, v in stats.items()}

    def run(self):
        print(f"Start Training on {self.cfg.device}...")
        start_time = time.time()
        
        for epoch in range(1, self.cfg.epochs + 1):
            metrics = self.train_epoch(epoch)
            self.scheduler.step()
            
            # Log
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d} | Time {elapsed/60:.1f}m | "
                  f"Total {metrics['total']:.4f} | "
                  f"NLL {metrics['nll']:.3f} | KL {metrics['kl']:.3f} | "
                  f"Con {metrics['con']:.3f} | Eng {metrics['eng']:.3f}")
            
            # Save periodically
            if epoch % 5 == 0 or epoch == self.cfg.epochs:
                self.save_checkpoint(epoch, metrics)


if __name__ == "__main__":
    # Set Seeds
    torch.manual_seed(42)
    
    # Config
    cfg = PretrainConfig()
    
    # Run
    try:
        trainer = Trainer(cfg)
        trainer.run()
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        trainer.save_checkpoint(999, {})
    except Exception as e:
        print(f"Error occurred: {e}")
        # In a notebook, this helps see if data download failed
        import traceback
        traceback.print_exc()
