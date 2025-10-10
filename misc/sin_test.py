# freq_transformer_demo.py
from torch.cuda import is_available
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# ---------- 1) Synthetic dataset ----------
class SineMixDataset(Dataset):
    def __init__(
        self,
        n_samples=2000,
        seq_len=256,
        fs=128,
        n_channels=8,
        low=(6, 10),
        high=(18, 22),
        snr_db=0,
    ):
        self.seq_len, self.fs, self.n_channels = seq_len, fs, n_channels
        t = np.arange(seq_len) / fs
        X = np.zeros((n_samples, n_channels, seq_len), np.float32)
        y = np.zeros((n_samples,), np.int64)
        rng = np.random.default_rng(0)
        for i in range(n_samples):
            cls = rng.integers(0, 2)
            y[i] = cls
            f = rng.uniform(*(low if cls == 0 else high))
            for c in range(n_channels):
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.8, 1.2)
                sig = amp * np.sin(2 * np.pi * f * t + phase)
                sp = sig.var() + 1e-8
                noise = rng.normal(0, np.sqrt(sp), size=t.shape)  # ~0 dB SNR
                X[i, c] = (sig + noise).astype(np.float32)
        # Per-channel standardization over the dataset
        Xf = X.transpose(0, 2, 1).reshape(-1, n_channels)
        m, s = Xf.mean(0, keepdims=True), Xf.std(0, keepdims=True) + 1e-6
        X[:] = Xf.reshape(n_samples, seq_len, n_channels).transpose(0, 2, 1)
        X = ((X.transpose(0, 2, 1) - m) / s).transpose(0, 2, 1).astype(np.float32)
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ---------- 2) Positional encodings ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# ---------- 3) Tiny Transformer backbone ----------
class TinyTransformer(nn.Module):
    def __init__(
        self, d_model=64, nhead=4, dim_ff=128, nlayers=2, seq_len=256, drop=0.1
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=drop, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.pos = PositionalEncoding(d_model, seq_len)

    def forward(self, x):  # (B, T, D)
        return self.encoder(self.pos(x))


# ---------- 4) Two models ----------
class ModelA_RawTransformer(nn.Module):
    def __init__(self, C=8, D=64, seq_len=256, n_classes=2):
        super().__init__()
        self.inp = nn.Linear(C, D)
        self.backbone = TinyTransformer(
            D, nhead=4, dim_ff=128, nlayers=2, seq_len=seq_len
        )
        self.cls = nn.Linear(D, n_classes)

    def forward(self, x):  # x: (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.inp(x)  # (B, T, D)
        h = self.backbone(x)  # (B, T, D)
        return self.cls(h.mean(1)), h


class ModelB_ConvThenTransformer(nn.Module):
    def __init__(self, C=8, D=64, seq_len=256, n_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(C, 32, kernel_size=9, padding=4)
        self.bn = nn.BatchNorm1d(32)
        self.proj = nn.Linear(32, D)
        self.backbone = TinyTransformer(
            D, nhead=4, dim_ff=128, nlayers=2, seq_len=seq_len
        )
        self.cls = nn.Linear(D, n_classes)

    def forward(self, x):  # x: (B, C, T)
        z = F.gelu(self.bn(self.conv(x)))  # (B, 32, T)
        z = self.proj(z.transpose(1, 2))  # (B, T, D)
        h = self.backbone(z)  # (B, T, D)
        return self.cls(h.mean(1)), h


# ---------- 5) Train / eval ----------
def train(model, train_loader, val_loader, epochs=8, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    tr_hist, va_hist = [], []
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
        tr_loss = run / len(train_loader.dataset)
        # val
        model.eval()
        corr = tot = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)[0].argmax(1)
                corr += (pred == yb).sum().item()
                tot += yb.size(0)
        val_acc = corr / tot
        tr_hist.append(tr_loss)
        va_hist.append(val_acc)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_acc={val_acc:.3f}")
        if val_acc > best:
            best, best_state = val_acc, {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state:
        model.load_state_dict(best_state)
    return tr_hist, va_hist, best


def accuracy(model, loader, device="cpu"):
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            corr += (model(xb)[0].argmax(1) == yb).sum().item()
            tot += yb.size(0)
    return corr / tot


# ---------- 6) Run experiment ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    ds = SineMixDataset(n_samples=2000, seq_len=256, fs=128, n_channels=8, snr_db=0)
    n_train = int(0.7 * len(ds))
    n_val = int(0.15 * len(ds))
    n_test = len(ds) - n_train - n_val
    train_set, val_set, test_set = random_split(
        ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    print("\nModel A: Raw Transformer")
    modelA = ModelA_RawTransformer(C=8, D=64, seq_len=256)
    a_tr, a_va, a_best = train(
        modelA, train_loader, val_loader, epochs=10, lr=1e-3, device=device
    )

    print("\nModel B: Conv → Transformer")
    modelB = ModelB_ConvThenTransformer(C=8, D=64, seq_len=256)
    b_tr, b_va, b_best = train(
        modelB, train_loader, val_loader, epochs=10, lr=1e-3, device=device
    )

    a_test = accuracy(modelA, test_loader, device=device)
    b_test = accuracy(modelB, test_loader, device=device)
    print(f"\nTest accuracy — Model A (raw Transformer): {a_test:.3f}")
    print(f"Test accuracy — Model B (conv → Transformer): {b_test:.3f}")

    # Curves
    plt.figure()
    plt.plot(a_tr, label="A: train_loss")
    plt.plot(b_tr, label="B: train_loss")
    plt.legend()
    plt.title("Training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(a_va, label="A: val_acc")
    plt.plot(b_va, label="B: val_acc")
    plt.legend()
    plt.title("Validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()
    plt.show()

    # ---------- 7) Attention map (Model A, layer 1, head 0) ----------
    enc0 = modelA.backbone.encoder.layers[0]
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            # One high-freq example if available
            mask = yb == 1
            x1 = xb[mask][0:1] if mask.any() else xb[0:1]
            # Tokens
            tok = modelA.inp(x1.transpose(1, 2))  # (1,T,D)
            tok = modelA.backbone.pos(tok)
            y = enc0.norm1(tok)
            out, attn = enc0.self_attn(
                y, y, y, need_weights=True, average_attn_weights=False
            )
            attn = attn.squeeze(0)[0].detach().cpu().numpy()  # head 0, (T,T)
            break

    plt.figure()
    plt.imshow(attn, aspect="auto", origin="lower", interpolation="nearest")
    plt.title("Model A — Layer 1, Head 0 attention")
    plt.xlabel("Key timestep")
    plt.ylabel("Query timestep")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print("\nObservation:")
    print("- You should see diagonal ‘bands’ at constant offsets in the attention map,")
    print(
        "  indicating the model has latched onto periodic structure (i.e., frequency)."
    )
