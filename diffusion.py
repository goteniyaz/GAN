#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, math, numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data.txt",
                    help="Path to dataset. Last column = class; others = AP features.")
parser.add_argument("--sep", type=str, default=",",
                    help="CSV separator (default ','). For whitespace/tab use --sep \"\\s+|\\t\" and engine='python'.")
parser.add_argument("--engine", type=str, default="c",
                    help="pandas read_csv engine (default 'c'; use 'python' if sep is a regex).")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--steps", type=int, default=100, help="diffusion steps T")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()  # use [] if running inside a notebook

# ---------------------------
# Repro
# ---------------------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ---------------------------
# Device (CUDA -> MPS -> CPU)
# ---------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ---------------------------
# Load data: features = AP columns, y = last col (classes)
# ---------------------------
df = pd.read_csv(args.data, header=None, sep=args.sep, engine=args.engine)
X = df.iloc[:, :-1].values.astype(np.float32)   # APs
y_raw = df.iloc[:, -1].values                   # classes (maybe non-int)

# Map labels to 0..C-1 (works even if input labels are strings or 1..C)
classes, y = np.unique(y_raw, return_inverse=True)
y = y.astype(np.int64)
num_classes = len(classes)
feat_dim = X.shape[1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=args.seed, stratify=y
)

# Standardize AP features (fit on train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# Tensors + loaders
Xtr = torch.from_numpy(X_train)
ytr = torch.from_numpy(y_train)
Xte = torch.from_numpy(X_test)
yte = torch.from_numpy(y_test)

pin = True if device.type == "cuda" else False
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True, pin_memory=pin)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch_size, shuffle=False, pin_memory=pin)

# ---------------------------
# Cosine-ish noise schedule
# ---------------------------
def beta_t(t, T):
    # simple cosine schedule in [0,1]
    return (1.0 - math.cos(math.pi * t / T)) / 2.0

# Precompute useful arrays on device
T = args.steps
betas = torch.tensor([beta_t(t, T) for t in range(T)], dtype=torch.float32, device=device)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)  # product_{s<=t} alpha_s

# ---------------------------
# Time embedding (sin/cos)
# ---------------------------
def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device) / max(half-1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb

# ---------------------------
# Simple MLP "U-Net" for tabular with class conditioning
# ---------------------------
class TabularCond(nn.Module):
    def __init__(self, x_dim, num_classes, time_dim=128, hidden=[512, 256, 128]):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, hidden[0])
        )
        self.class_emb = nn.Embedding(num_classes, hidden[0])

        in_dim = x_dim + hidden[0] + hidden[0]  # x + time + class
        self.blocks = nn.ModuleList()
        dims = [in_dim] + hidden
        for i in range(len(hidden)):
            self.blocks.append(nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.SiLU()))
        self.out = nn.Linear(hidden[-1], x_dim)

    def forward(self, x, y, t):
        # y: (B,) class indices
        # t: (B,) timesteps
        t_emb = get_timestep_embedding(t, self.time_dim)
        t_h = self.time_mlp(t_emb)
        c_h = self.class_emb(y)
        h = torch.cat([x, t_h, c_h], dim=1)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)  # predict noise ε

# Wrap as diffusion ε-predictor
class DiffusionEps(nn.Module):
    def __init__(self, x_dim, num_classes, time_dim=128, hidden=[512,256,128]):
        super().__init__()
        self.net = TabularCond(x_dim, num_classes, time_dim, hidden)
    def forward(self, x_t, y, t):
        return self.net(x_t, y, t)

model = DiffusionEps(feat_dim, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
mse = nn.MSELoss()

# ---------------------------
# Forward diffusion: sample x_t, target noise ε
# q(x_t | x_0) = N( sqrt(ᾱ_t) x_0, (1-ᾱ_t) I )
# ---------------------------
def q_sample(x0, t, noise=None):
    """
    x0: (B,D)
    t : (B,) long
    """
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar = alphas_bar[t].unsqueeze(1)  # (B,1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

# ---------------------------
# Training
# ---------------------------
def train(model, loader, epochs=50):
    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for x0, y in loader:
            x0 = x0.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            B = x0.size(0)
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            x_t, noise = q_sample(x0, t)
            pred = model(x_t, y, t)
            loss = mse(pred, noise)  # denoising score matching loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * B
        print(f"Epoch {ep:03d}/{epochs} | train diffusion loss (≈NLL proxy): {total/len(loader.dataset):.4f}")

def avg_diffusion_loss(model, loader):
    """Average denoising loss on a dataset."""
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for x0, y in loader:
            x0 = x0.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            B = x0.size(0)
            # Monte Carlo over time steps for fair average (T samples per batch)
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            x_t, noise = q_sample(x0, t)
            pred = model(x_t, y, t)
            loss = mse(pred, noise)
            total += loss.item() * B
            n += B
    return total / max(n, 1)

def generative_classify(model, X, steps_for_energy=64):
    """
    Energy-based generative classifier:
    For each sample x, for each class c, compute diffusion loss L_c(x)
    (avg over random timesteps), pick argmin_c L_c(x).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, in DataLoader(TensorDataset(X), batch_size=512, shuffle=False, pin_memory=pin):
            xb = xb[0].to(device, non_blocking=True)
            B = xb.size(0)
            # Expand to (B, C, D)
            xb_expand = xb.unsqueeze(1).expand(B, num_classes, feat_dim).reshape(-1, feat_dim)
            y_all = torch.arange(num_classes, device=device).repeat_interleave(B)
            # Estimate energy per (x, class)
            E = torch.zeros(B * num_classes, device=device)
            K = steps_for_energy
            for _ in range(K):
                t = torch.randint(0, T, (B * num_classes,), device=device, dtype=torch.long)
                x_t, noise = q_sample(xb_expand, t)       # treat xb as x0 to derive (x_t, noise)
                pred = model(x_t, y_all, t)
                E += torch.mean((pred - noise) ** 2, dim=1)
            E = E / K
            # Argmin over classes
            E = E.view(B, num_classes)
            yhat = torch.argmin(E, dim=1)
            preds.append(yhat.cpu())
    return torch.cat(preds, dim=0).numpy()

# ---------------------------
# Run
# ---------------------------
print(f"Device: {device} | Classes: {num_classes} | Feature dim: {feat_dim} | Steps: {T}")
train(model, train_loader, epochs=args.epochs)

test_loss_proxy = avg_diffusion_loss(model, test_loader)
y_pred = generative_classify(model, Xte, steps_for_energy=64)
acc = (y_pred == yte.numpy()).mean()

print("\n=== Evaluation ===")
print(f"Accuracy: {acc:.4f}")
print(f"Avg diffusion loss (NLL proxy, lower is better): {test_loss_proxy:.4f}")
