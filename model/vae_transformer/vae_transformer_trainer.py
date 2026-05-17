from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..shared.config import (
    BATCH_SIZE, LEARNING_RATE, RANDOM_SEED,
    TRANS_LATENT_DIM, TRANS_D_MODEL, TRANS_NUM_HEADS,
    TRANS_FF_DIM, TRANS_NUM_LAYERS, TRANS_DROPOUT,
    TRANS_BETA, TRANS_EPOCHS,
    NUM_SAMPLES_AT_INFERENCE,
)
from ..shared.tokenizer import load_dataset, records_to_arrays
from ..shared.validator import evaluate_predictions, pick_best_sample

from .vae_transformer_model import VAETransformer

SAVE_DIR    = Path(__file__).parent / "weights"
BETA_WARMUP = 20


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_weighted_sampler(conditions: np.ndarray) -> WeightedRandomSampler:
    decade_idx = conditions[:, 3]
    counts     = np.bincount(decade_idx, minlength=41).astype(np.float32)
    counts     = np.where(counts == 0, 1, counts)
    weights    = (1.0 / counts)[decade_idx]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float),
        num_samples=len(weights),
        replacement=True,
    )


# Training entry point 

def train(
    data_path:  str,
    epochs:     int   = TRANS_EPOCHS,
    batch_size: int   = BATCH_SIZE,
    d_model:    int   = TRANS_D_MODEL,
    n_heads:    int   = TRANS_NUM_HEADS,
    ff_dim:     int   = TRANS_FF_DIM,
    n_layers:   int   = TRANS_NUM_LAYERS,
    latent_dim: int   = TRANS_LATENT_DIM,
    dropout:    float = TRANS_DROPOUT,
    beta:       float = TRANS_BETA,
    lr:         float = LEARNING_RATE,
    seed:       int   = RANDOM_SEED,
) -> None:
    set_seeds(seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = device.type == "cuda"
    print(f"Using device: {device}  |  Mixed precision: {use_amp}")

    print("Loading dataset …")
    records           = load_dataset(data_path)
    conditions, dates = records_to_arrays(records)

    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(records))
    split = int(0.9 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]

    cond_train = conditions[tr_idx]
    date_train = dates[tr_idx]       # normalised [0,1]
    cond_val   = conditions[val_idx]
    date_val   = dates[val_idx]

    print(f"  Train: {len(tr_idx):,}  |  Val: {len(val_idx):,}")

    # Denorm to get raw date values for training (model expects raw)
    date_offsets = torch.tensor([1.0, 1.0, 1800.0])
    date_ranges  = torch.tensor([30.0, 11.0, 400.0])
    date_raw_train = (torch.tensor(date_train, dtype=torch.float32) * date_ranges + date_offsets)

    dataset = TensorDataset(
        torch.tensor(cond_train, dtype=torch.long),
        date_raw_train,
    )
    sampler      = make_weighted_sampler(cond_train)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        drop_last=True, num_workers=0,
    )

    # Model 
    model = VAETransformer(
        d_model=d_model, n_heads=n_heads, ff_dim=ff_dim,
        n_layers=n_layers, latent_dim=latent_dim, dropout=dropout, beta=beta,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    history  = {"total_loss": [], "recon_loss": [], "kl_loss": [], "val_accuracy": []}
    best_acc = -1.0

    # Training loop 
    for epoch in range(1, epochs + 1):
        model.train()
        current_beta = min(beta, beta * epoch / BETA_WARMUP)

        ep_total = ep_recon = ep_kl = 0.0
        n_batches = 0
        t0 = time.time()

        for cond_batch, date_batch in train_loader:
            cond_batch = cond_batch.to(device)
            date_batch = date_batch.to(device)

            optimiser.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                total, recon, kl = model.compute_loss(
                    date_batch, cond_batch, beta=current_beta
                )

            scaler.scale(total).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            ep_total += total.item()
            ep_recon += recon.item()
            ep_kl    += kl.item()
            n_batches += 1

        avg_total = ep_total / n_batches
        avg_recon = ep_recon / n_batches
        avg_kl    = ep_kl    / n_batches
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        #  Validation 
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_preds  = []
            cond_val_t = torch.tensor(cond_val, dtype=torch.long, device=device)

            with torch.no_grad():
                for i in range(0, len(cond_val), batch_size):
                    c_batch  = cond_val_t[i : i + batch_size]
                    samples  = model.sample(c_batch, num_samples=NUM_SAMPLES_AT_INFERENCE)
                    samps_np = samples.cpu().numpy()             # (batch, S, 3)
                    for j, samps in enumerate(samps_np):
                        best = pick_best_sample(samps, cond_val[i + j])
                        val_preds.append(best)

            val_arr = np.array(val_preds, dtype=np.float32)
            results = evaluate_predictions(cond_val, val_arr, verbose=False)
            history["val_accuracy"].append(results["all"])

            if results["all"] > best_acc:
                best_acc = results["all"]
                torch.save(model.state_dict(), str(SAVE_DIR / "vaetrans_best.pt"))

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {avg_total:.4f} (R:{avg_recon:.4f} KL:{avg_kl:.4f}) | "
                f"β={current_beta:.2f} | Val: {results['all']*100:.1f}%  "
                f"[{time.time()-t0:.1f}s]"
            )
        else:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {avg_total:.4f} (R:{avg_recon:.4f} KL:{avg_kl:.4f}) | "
                f"β={current_beta:.2f}  [{time.time()-t0:.1f}s]"
            )

    torch.save(model.state_dict(), str(SAVE_DIR / "vaetrans_final.pt"))
    np.save(str(SAVE_DIR / "training_history.npy"), history)
    print(f"\nBest val accuracy: {best_acc*100:.1f}%  →  {SAVE_DIR}")


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "../../data/data.txt"
    train(data_path=data_file)