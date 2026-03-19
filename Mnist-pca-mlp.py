"""
MNIST: PCA + MLP vs Raw MLP
Compares training time and accuracy with and without PCA dimensionality reduction.
Run: pip install torch scikit-learn numpy matplotlib
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Config ──────────────────────────────────────────────────────────────────
PCA_COMPONENTS   = 50       # target dims after PCA
VARIANCE_TARGET  = 0.95     # for reporting retained variance
HIDDEN_SIZE      = 128
EPOCHS           = 15
BATCH_SIZE       = 64
LR               = 1e-3
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}\n")

# ── 1. Load MNIST ────────────────────────────────────────────────────────────
print("Loading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y  = mnist.data.astype(np.float32), mnist.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# ── 2. Normalize ─────────────────────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. PCA ───────────────────────────────────────────────────────────────────
print(f"Fitting PCA (n_components={PCA_COMPONENTS})...")
pca           = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_train_pca   = pca.fit_transform(X_train_sc).astype(np.float32)
X_test_pca    = pca.transform(X_test_sc).astype(np.float32)

retained_var  = pca.explained_variance_ratio_.sum()
dim_reduction = (1 - PCA_COMPONENTS / X_train.shape[1]) * 100
print(f"  Retained variance : {retained_var:.4f} ({retained_var*100:.2f}%)")
print(f"  Dim reduction     : {X_train.shape[1]} → {PCA_COMPONENTS} ({dim_reduction:.1f}% reduction)\n")

# ── 4. MLP definition ────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

def make_loaders(X_tr, X_te, y_tr, y_te):
    tr = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                    batch_size=BATCH_SIZE, shuffle=True)
    te = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
                    batch_size=256)
    return tr, te

def train_and_eval(X_tr, X_te, y_tr, y_te, label):
    model     = MLP(X_tr.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    tr_loader, te_loader = make_loaders(X_tr, X_te, y_tr, y_te)

    print(f"── Training: {label} (input dim={X_tr.shape[1]}) ──")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2}/{EPOCHS}  loss={total_loss/len(tr_loader):.4f}")

    elapsed = time.time() - t0

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds   = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

    acc = correct / total
    print(f"  Test accuracy : {acc*100:.2f}%")
    print(f"  Training time : {elapsed:.1f}s\n")
    return acc, elapsed

# ── 5. Run both experiments ──────────────────────────────────────────────────
acc_raw, time_raw = train_and_eval(
    X_train_sc.astype(np.float32), X_test_sc.astype(np.float32),
    y_train, y_test,
    "Raw (784-dim)"
)

acc_pca, time_pca = train_and_eval(
    X_train_pca, X_test_pca,
    y_train, y_test,
    f"PCA ({PCA_COMPONENTS}-dim)"
)

# ── 6. Summary ───────────────────────────────────────────────────────────────
print("=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"{'':30s} {'Raw':>10} {'PCA':>10}")
print(f"{'Input dimensions':30s} {'784':>10} {str(PCA_COMPONENTS):>10}")
print(f"{'Retained variance':30s} {'100.00%':>10} {retained_var*100:>9.2f}%")
print(f"{'Test accuracy':30s} {acc_raw*100:>9.2f}% {acc_pca*100:>9.2f}%")
print(f"{'Training time (s)':30s} {time_raw:>10.1f} {time_pca:>10.1f}")
print(f"{'Speedup':30s} {'':>10} {time_raw/time_pca:>9.2f}x")
print("=" * 50)

# ── 7. Plot: explained variance curve ───────────────────────────────────────
pca_full = PCA(random_state=42).fit(X_train_sc)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(cumvar, linewidth=2)
plt.axhline(VARIANCE_TARGET, color="red",  linestyle="--", label=f"{VARIANCE_TARGET*100:.0f}% variance")
plt.axvline(PCA_COMPONENTS,  color="green", linestyle="--", label=f"n={PCA_COMPONENTS} components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Cumulative Explained Variance on MNIST")
plt.legend()
plt.tight_layout()
plt.savefig("pca_variance_curve.png", dpi=150)
plt.show()
print("Saved pca_variance_curve.png")
