# freq_transformer_demo.py
from math import gcd
from functools import reduce

from einops import rearrange
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.components.rope import RotaryEmbedding
from src.montagenet import (
    DataConfig,
    KernelFactoryType,
    TemporalAttentionBlock,
    ContinuousSignalEmbedder,
)


# ---------- 1) Synthetic dataset ----------


def lcm2(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a) // gcd(a, b) * abs(b)


def lcmN(*nums: int) -> int:
    return reduce(lcm2, nums, 1)


class SineMixDataset(Dataset):
    def __init__(
        self,
        n_samples,
        seq_len_sec,
        sampling_rates,
        n_channels,
        low,
        high,
        snr_db,
    ):
        self.seq_len_sec, self.n_channels = seq_len_sec, n_channels
        ts = [np.arange(int(sr * seq_len_sec)) for sr in sampling_rates]
        max_len = max([len(t) for t in ts])
        X = np.zeros((n_samples, n_channels, max_len), np.float32)
        M = np.zeros((n_samples, max_len), np.int32)
        SR = np.zeros((n_samples,), np.int32)
        lcm = lcmN(*sampling_rates)
        seq_positions = np.zeros((n_samples, max_len), np.int64)
        y = np.zeros((n_samples,), np.int64)
        rng = np.random.default_rng(0)
        for i in range(n_samples):
            cls = rng.integers(0, 2)
            sr_i = rng.integers(len(sampling_rates))
            sr = sampling_rates[sr_i]
            t = ts[sr_i]
            y[i] = cls
            M[i, : len(t)] = 1
            SR[i] = sr
            seq_positions[i, : len(t)] = np.arange(0, int(lcm * seq_len_sec), lcm // sr)
            f = rng.uniform(*(low if cls == 0 else high))
            snr_lin = float("inf") if snr_db is None else 10 ** (snr_db / 10.0)
            for c in range(n_channels):
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.8, 1.2)
                sig = amp * np.sin(2 * np.pi * f * (t / sr) + phase)
                sp = sig.var() + 1e-8
                if np.isinf(snr_lin):
                    noise = 0.0
                else:
                    noise_std = np.sqrt(sp / snr_lin)
                    noise = rng.normal(0.0, noise_std, size=t.shape)
                X[i, c, 0 : len(sig)] = (sig + noise).astype(np.float32)
        # Per-channel standardization over valid (masked) timesteps only
        mask = M[:, None, :].astype(np.float32)
        count = np.maximum(mask.sum(axis=(0, 2), keepdims=True), 1.0)
        mean = (X * mask).sum(axis=(0, 2), keepdims=True) / count
        var = (((X - mean) ** 2) * mask).sum(axis=(0, 2), keepdims=True) / count
        std = np.sqrt(var) + 1e-6
        X = ((X - mean) / std).astype(np.float32)
        self.X, self.SR, self.L, self.M, self.y = X, SR, seq_positions, M, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i]),
            torch.tensor(self.M[i], dtype=torch.bool),
            torch.tensor(self.L[i]),
            torch.tensor(self.SR[i]),
            torch.tensor(self.y[i]),
        )


# ---------- 3) Tiny Transformer backbone ----------


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, nlayers, drop):
        super().__init__()
        rotary_embedding = RotaryEmbedding(dim=8)
        self.layers = nn.ModuleList(
            [
                TemporalAttentionBlock(d_model, nhead, dim_ff, rotary_embedding, drop)
                for _ in range(nlayers)
            ]
        )

    def forward(self, x, mask, seq_pos):  # (B, T, D)
        for layer in self.layers:
            x = layer(x, mask, seq_pos)
        return x


# ---------- 4) Two models ----------
class ModelA_RawTransformer(nn.Module):
    def __init__(
        self,
        data_config,
        d_model,
        n_classes,
        nhead,
        dim_ff,
        nlayers,
        dropout,
    ):
        super().__init__()
        n_channels = data_config.channel_counts[0]
        self.inp = nn.Linear(n_channels, d_model * n_channels)
        self.backbone = Transformer(
            d_model,
            nhead,
            dim_ff,
            nlayers,
            dropout,
        )
        self.cls = nn.Linear(n_channels * d_model, n_classes)

    def forward(self, x, sr, mask, seq_pos):
        B, C, T = x.shape
        n_samples = mask.sum(dim=1)
        expanded_mask = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        seq_pos = seq_pos.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        x = x.transpose(1, 2)
        x = self.inp(x)  # (B, T, DC)
        h = self.backbone(
            rearrange(x, "B T (D C) -> (B C) T D", B=B, T=T, C=C),
            expanded_mask,
            seq_pos,
        )
        h = rearrange(h, "(B C) T D -> B T (D C)", B=B, C=C, T=T)
        h = (h * mask.unsqueeze(-1)).sum(dim=1) / n_samples.unsqueeze(1)
        return self.cls(h), h


class ModelB_MorletConvThenTransformer(nn.Module):
    def __init__(
        self,
        data_config,
        d_model,
        n_classes,
        nhead,
        dim_ff,
        nlayers,
        dropout,
    ):
        super().__init__()
        n_channels = data_config.channel_counts[0]
        self.conv = ContinuousSignalEmbedder(data_config, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.backbone = Transformer(d_model, nhead, dim_ff, nlayers, dropout)
        self.cls = nn.Linear(d_model * n_channels, n_classes)

    def forward(self, x, sr, mask, seq_pos):  # x: (B, C, T)
        B, C, T = x.shape
        n_samples = mask.sum(dim=1)
        expanded_mask = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        seq_pos = seq_pos.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        channel_counts = torch.tensor([C] * B)
        x = rearrange(x, "B C T -> (B C) T", B=B, C=C)
        z = self.conv(
            x,
            channel_counts,
            sr.tolist(),
        )
        h = self.backbone(
            rearrange(z, "B C T D -> (B C) T D"),
            expanded_mask,
            seq_pos,
        )  # (BC, T, D)
        h = rearrange(h, "(B C) T D -> B T (D C)", B=B, C=C, T=T)
        h = (h * mask.unsqueeze(-1)).sum(dim=1) / n_samples.unsqueeze(1)
        return self.cls(h), h


class ModelC_SpectralConvThenTransformer(nn.Module):
    def __init__(
        self,
        data_config,
        d_model,
        n_classes,
        nhead,
        dim_ff,
        nlayers,
        dropout,
    ):
        super().__init__()
        n_channels = data_config.channel_counts[0]
        self.conv = ContinuousSignalEmbedder(
            data_config, d_model, kernel_factory_type=KernelFactoryType.SPECTRUM
        )
        self.proj = nn.Linear(d_model, d_model)
        self.backbone = Transformer(d_model, nhead, dim_ff, nlayers, dropout)
        self.cls = nn.Linear(d_model * n_channels, n_classes)

    def forward(self, x, sr, mask, seq_pos):  # x: (B, C, T)
        B, C, T = x.shape
        n_samples = mask.sum(dim=1)
        expanded_mask = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        seq_pos = seq_pos.unsqueeze(1).expand(-1, C, -1).reshape(B * C, T)
        channel_counts = torch.tensor([C] * B)
        x = rearrange(x, "B C T -> (B C) T", B=B, C=C)
        z = self.conv(
            x,
            channel_counts,
            sr.tolist(),
        )
        h = self.backbone(
            rearrange(z, "B C T D -> (B C) T D"),
            expanded_mask,
            seq_pos,
        )  # (BC, T, D)
        h = rearrange(h, "(B C) T D -> B T (D C)", B=B, C=C, T=T)
        h = (h * mask.unsqueeze(-1)).sum(dim=1) / n_samples.unsqueeze(1)
        return self.cls(h), h


# ---------- 5) Train / eval ----------
def train(model, train_loader, val_loader, epochs, lr, weight_decay, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    best, best_state = 0.0, None
    tr_hist, va_hist = [], []
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for xb, mb, lb, sb, yb in train_loader:
            xb, mb, lb, sb, yb = (
                xb.to(device),
                mb.to(device),
                lb.to(device),
                sb.to(device),
                yb.to(device),
            )
            opt.zero_grad()
            logits, _ = model(xb, sb, mb, lb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
        tr_loss = run / len(train_loader.dataset)
        # val
        model.eval()
        corr = tot = 0
        with torch.no_grad():
            for xb, mb, lb, sb, yb in val_loader:
                xb, mb, lb, sb, yb = (
                    xb.to(device),
                    mb.to(device),
                    lb.to(device),
                    sb.to(device),
                    yb.to(device),
                )
                pred = model(xb, sb, mb, lb)[0].argmax(1)
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


def accuracy(model, loader, device):
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for xb, mb, lb, sb, yb in loader:
            xb, mb, lb, sb, yb = (
                xb.to(device),
                mb.to(device),
                lb.to(device),
                sb.to(device),
                yb.to(device),
            )
            corr += (model(xb, sb, mb, lb)[0].argmax(1) == yb).sum().item()
            tot += yb.size(0)
    return corr / tot


# ---------- 6) Run experiment ----------
if __name__ == "__main__":
    MANUAL_SEED = 0
    NUMPY_SEED = 0
    CHANNEL_COUNTS = [8]
    SAMPLING_RATES = [128, 180]
    FREQ_LOW_BAND = (6, 10)
    FREQ_HIGH_BAND = (18, 22)
    KERNEL_SECONDS = 0.5
    SEQUENCE_LENGTH_SECONDS = 1.0
    DATASET_NUM_SAMPLES = 2000
    DATASET_FS = 128
    DATASET_NUM_CHANNELS = 8
    DATASET_SNR_DB = 0
    TRAIN_FRACTION = 0.7
    VAL_FRACTION = 0.15
    TRAIN_BATCH_SIZE = 128
    VAL_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    NUM_WORKERS = 0
    MODEL_D_MODEL = 32
    MODEL_NUM_CLASSES = 2
    MODEL_NHEAD = 4
    MODEL_DIM_FF = 64
    MODEL_NLAYERS = 2
    MODEL_DROPOUT = 0.1
    TRAIN_EPOCHS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01

    torch.manual_seed(MANUAL_SEED)
    np.random.seed(NUMPY_SEED)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    data_config = DataConfig(
        channel_counts=CHANNEL_COUNTS,
        sampling_rates=SAMPLING_RATES,
        f_min=FREQ_LOW_BAND[0],
        f_max=FREQ_HIGH_BAND[1],
        kernel_sec=KERNEL_SECONDS,
        sequence_length_seconds=SEQUENCE_LENGTH_SECONDS,
    )

    ds = SineMixDataset(
        n_samples=DATASET_NUM_SAMPLES,
        seq_len_sec=int(SEQUENCE_LENGTH_SECONDS),
        sampling_rates=SAMPLING_RATES,
        n_channels=DATASET_NUM_CHANNELS,
        low=FREQ_LOW_BAND,
        high=FREQ_HIGH_BAND,
        snr_db=DATASET_SNR_DB,
    )
    n_train = int(TRAIN_FRACTION * len(ds))
    n_val = int(VAL_FRACTION * len(ds))
    n_test = len(ds) - n_train - n_val
    train_set, val_set, test_set = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(MANUAL_SEED),
    )
    train_loader = DataLoader(
        train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # print("\nModel A: Raw Transformer")
    # modelA = ModelA_RawTransformer(
    #     data_config=data_config,
    #     d_model=MODEL_D_MODEL,
    #     n_classes=MODEL_NUM_CLASSES,
    #     nhead=MODEL_NHEAD,
    #     dim_ff=MODEL_DIM_FF,
    #     nlayers=MODEL_NLAYERS,
    #     dropout=MODEL_DROPOUT,
    # )
    # a_tr, a_va, a_best = train(
    #     modelA,
    #     train_loader,
    #     val_loader,
    #     epochs=TRAIN_EPOCHS,
    #     lr=LEARNING_RATE,
    #     weight_decay=WEIGHT_DECAY,
    #     device=device,
    # )

    # print("\nModel B: Morlet Conv → Transformer")
    # modelB = ModelB_MorletConvThenTransformer(
    #     data_config=data_config,
    #     d_model=MODEL_D_MODEL,
    #     n_classes=MODEL_NUM_CLASSES,
    #     nhead=MODEL_NHEAD,
    #     dim_ff=MODEL_DIM_FF,
    #     nlayers=MODEL_NLAYERS,
    #     dropout=MODEL_DROPOUT,
    # )
    # b_tr, b_va, b_best = train(
    #     modelB,
    #     train_loader,
    #     val_loader,
    #     epochs=TRAIN_EPOCHS,
    #     lr=LEARNING_RATE,
    #     weight_decay=WEIGHT_DECAY,
    #     device=device,
    # )
    print("\nModel C: Spectrum Conv → Transformer")
    modelC = ModelC_SpectralConvThenTransformer(
        data_config=data_config,
        d_model=MODEL_D_MODEL,
        n_classes=MODEL_NUM_CLASSES,
        nhead=MODEL_NHEAD,
        dim_ff=MODEL_DIM_FF,
        nlayers=MODEL_NLAYERS,
        dropout=MODEL_DROPOUT,
    )
    c_tr, c_va, c_best = train(
        modelC,
        train_loader,
        val_loader,
        epochs=TRAIN_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    # a_test = accuracy(modelA, test_loader, device=device)
    # b_test = accuracy(modelB, test_loader, device=device)
    c_test = accuracy(modelC, test_loader, device=device)
    # print(f"\nTest accuracy — Model A (raw Transformer): {a_test:.3f}")
    # print(f"Test accuracy — Model B (morlet conv → Transformer): {b_test:.3f}")
    print(f"Test accuracy — Model B (spectrum conv → Transformer): {c_test:.3f}")

    # Curves
    plt.figure()
    # plt.plot(a_tr, label="A: train_loss")
    # plt.plot(b_tr, label="B: train_loss")
    plt.plot(c_tr, label="C: train_loss")
    plt.legend()
    plt.title("Training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()

    plt.figure()
    # plt.plot(a_va, label="A: val_acc")
    # plt.plot(b_va, label="B: val_acc")
    plt.plot(c_va, label="C: val_acc")
    plt.legend()
    plt.title("Validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()
    plt.show()

    # # ---------- 7) Attention map (Model A, layer 1, head 0) ----------
    # enc0 = modelA.backbone.encoder.layers[0]
    # with torch.no_grad():
    #     for xb, yb in test_loader:
    #         xb = xb.to(device)
    #         # One high-freq example if available
    #         mask = yb == 1
    #         x1 = xb[mask][0:1] if mask.any() else xb[0:1]
    #         # Tokens
    #         tok = modelA.inp(x1.transpose(1, 2))  # (1,T,D)
    #         tok = modelA.backbone.pos(tok)
    #         y = enc0.norm1(tok)
    #         out, attn = enc0.self_attn(
    #             y, y, y, need_weights=True, average_attn_weights=False
    #         )
    #         attn = attn.squeeze(0)[0].detach().cpu().numpy()  # head 0, (T,T)
    # break

    # plt.figure()
    # plt.imshow(attn, aspect="auto", origin="lower", interpolation="nearest")
    # plt.title("Model A — Layer 1, Head 0 attention")
    # plt.xlabel("Key timestep")
    # plt.ylabel("Query timestep")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # print("\nObservation:")
    # print("- You should see diagonal ‘bands’ at constant offsets in the attention map,")
    # print(
    #     "  indicating the model has latched onto periodic structure (i.e., frequency)."
    # )
