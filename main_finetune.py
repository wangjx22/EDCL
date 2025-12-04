import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9, QM7b, MoleculeNet
import torch_geometric.transforms as T
from torch_geometric.data import Data

try:
    
    from torch_geometric.transforms import Add3dCoords 
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class FinetuneConfig:
    def __init__(self):
        self.hidden_dim = 128
        self.batch_size = 32
        self.lr = 5e-4
        self.epochs = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_ckpt = "./checkpoints/ckpt_50.pt"
        
        # --- Dataset Configuration ---
        # Options: 'QM9', 'QM7', 'ESOL', 'Lipophilicity', 'FreeSolv', etc.
        self.dataset_name = 'QM9' 
        
        # For QM9: choose target index (0-18). 
        # e.g., 7=HOMO, 8=LUMO, 10=Gap, 12=U0
        self.target_idx = 7 
        
        # Split ratio
        self.split = [0.8, 0.1, 0.1] # Train/Val/Test

class TargetSelector(object):
    """
    QM9 有 19 个回归目标。我们通常一次只训练一个。
    该 Transform 提取指定列作为 y。
    """
    def __init__(self, target_idx):
        self.target_idx = target_idx

    def __call__(self, data):
        # data.y shape in QM9 is [1, 19]
        # We select one column and keep shape [1, 1]
        if data.y is not None and data.y.shape[-1] > 1:
            data.y = data.y[:, self.target_idx:self.target_idx+1]
        return data

class LabelNormalizer:
    """
    回归任务必备：对 Label 进行标准化 (y - mean) / std
    在训练开始前根据训练集统计量计算。
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def fit(self, loader):
        # 遍历 DataLoader 计算 mean 和 std
        all_y = []
        for batch in loader:
            all_y.append(batch.y)
        all_y = torch.cat(all_y, dim=0)
        self.mean = all_y.mean().item()
        self.std = all_y.std().item()
        print(f"Label Normalization Stats: Mean={self.mean:.4f}, Std={self.std:.4f}")

    def normalize(self, tensor):
        if self.mean is None: return tensor
        return (tensor - self.mean) / (self.std + 1e-6)

    def denormalize(self, tensor):
        if self.mean is None: return tensor
        return tensor * (self.std + 1e-6) + self.mean


def get_dataset(name, root='./data'):
    """
    根据名称加载数据集，并确保拥有 3D 坐标 (pos)。
    """
    path = f"{root}/{name}"
    
    if name == 'QM9':
        # QM9: Native 3D, 19 regression targets
        dataset = QM9(root=path)
        return dataset, 'regression', 19 # 19 targets total, usually slice later

    elif name == 'QM7':
        # QM7b: Native 3D, 14 regression targets
        dataset = QM7b(root=path)
        return dataset, 'regression', 14

    elif name in ['ESOL', 'FreeSolv', 'Lipophilicity', 'HIV', 'BACE']:
        # MoleculeNet: Usually 2D SMILES based. Needs RDKit for 3D.
        pre_transform = None
        if HAS_RDKIT:
            print(f"Using RDKit to generate 3D conformers for {name}...")
            # Add3dCoords 会尝试生成 3D pos，如果失败则可能产生 NaN 或报错，需过滤
            pre_transform = Add3dCoords() 
        else:
            print("Warning: RDKit not found. MoleculeNet datasets usually lack 3D coordinates.")
            print("Equiformer will FAIL without 'pos'. Please install rdkit.")
        
        dataset = MoleculeNet(root=path, name=name, pre_transform=pre_transform)
        
        # 过滤掉生成 3D 失败的样本 (pos is None)
        valid_indices = []
        for i in range(len(dataset)):
            if dataset[i].pos is not None:
                valid_indices.append(i)
        
        if len(valid_indices) < len(dataset):
            print(f"Filtered {len(dataset) - len(valid_indices)} molecules due to 3D generation failure.")
            dataset = dataset.index_select(valid_indices)
            
        # Determine task type
        task_type = 'regression' if name in ['ESOL', 'FreeSolv', 'Lipophilicity'] else 'classification'
        num_tasks = dataset.num_classes if task_type == 'classification' else dataset.num_tasks
        
        return dataset, task_type, num_tasks

    else:
        raise ValueError(f"Unknown dataset: {name}")



def run_finetuning_pipeline():
    cfg = FinetuneConfig()
    
    # 1. Load Dataset
    print(f"Loading {cfg.dataset_name}...")
    full_dataset, task_type, raw_num_tasks = get_dataset(cfg.dataset_name)
    
    # 2. Pre-processing for QM9 (Select specific target)
    if cfg.dataset_name == 'QM9':
        print(f"Selecting QM9 target index {cfg.target_idx}...")
        full_dataset.transform = TargetSelector(cfg.target_idx)
        num_tasks = 1
    else:
        num_tasks = raw_num_tasks

    # 3. Split Data
    # Random Split (Scaffold split is better for MoleculeNet, but Random is standard for QM9)
    dataset_size = len(full_dataset)
    train_size = int(cfg.split[0] * dataset_size)
    val_size = int(cfg.split[1] * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    # 4. Label Normalization (For Regression Only)
    normalizer = LabelNormalizer()
    if task_type == 'regression':
        print("Computing statistics for label normalization...")
        normalizer.fit(train_loader)

    # 5. Initialize Model
    # Need to define the model classes here or import them
    backbone = EquiformerBackbone(cfg.hidden_dim)
    # Update config with actual num_tasks
    cfg.num_tasks = num_tasks 
    model = DownstreamPredictor(backbone, cfg).to(cfg.device)
    
    # Load Pretrained
    # try: model = load_pretrained_weights(model, cfg.pretrained_ckpt, cfg.device)
    # except: pass

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    
    # Loss function
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss() # For binary classification like HIV/Tox21

    # 6. Training Loop
    print(f"Start Fine-tuning on {task_type} task...")
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(cfg.device)
            
            # Apply Normalization to targets (if regression)
            target = batch.y
            if task_type == 'regression':
                target = normalizer.normalize(target)
            
            # QM9/ESOL target shape is often [B, 1], flatten if necessary for specific losses
            if task_type == 'classification':
                target = target.float()
            
            optimizer.zero_grad()
            pred = model(batch)
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        val_metric = evaluate(model, val_loader, normalizer, task_type, cfg.device)
        
        if epoch % 5 == 0:
            metric_name = "MAE" if task_type == 'regression' else "Loss"
            print(f"Epoch {epoch}: Train Loss: {total_loss/len(train_loader):.4f} | Val {metric_name}: {val_metric:.4f}")

def evaluate(model, loader, normalizer, task_type, device):
    model.eval()
    total_error = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            
            if task_type == 'regression':
                # Reverse normalization for metric calculation (MAE)
                pred_real = normalizer.denormalize(pred)
                error = torch.abs(pred_real - batch.y).sum()
                total_count += batch.y.numel()
                total_error += error.item()
            else:
                # Simple Loss for classification demo (AUC is better for MolNet)
                target = batch.y.float()
                loss = nn.BCEWithLogitsLoss()(pred, target)
                total_error += loss.item() * batch.num_graphs
                total_count += batch.num_graphs
                
    return total_error / total_count


class EquiformerBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Linear(1, dim)
        # Mock Layers
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
    def forward(self, data):
        x = self.embedding(data.x.float())
        h_graph = torch.mean(x, dim=0, keepdim=True).repeat(data.num_graphs, 1) # Mock global pool
        return x, h_graph

class DownstreamPredictor(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.encoder = backbone
        self.head = nn.Linear(config.hidden_dim, getattr(config, 'num_tasks', 1))
    def forward(self, data):
        _, h_graph = self.encoder(data)
        return self.head(h_graph)

if __name__ == "__main__":
    # 注意：运行此代码需要下载数据集，可能会花费一些时间
    # 默认配置为 QM9
    run_finetuning_pipeline()
