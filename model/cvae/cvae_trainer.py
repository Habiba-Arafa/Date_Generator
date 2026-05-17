from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..shared.config import (
    BATCH_SIZE, LEARNING_RATE, RANDOM_SEED,
    CVAE_LATENT_DIM, CVAE_HIDDEN_DIM, CVAE_BETA, CVAE_EPOCHS,
    NUM_SAMPLES_AT_INFERENCE,
)
from ..shared.tokenizer import load_dataset, records_to_arrays
from ..shared.validator import evaluate_predictions, pick_best_sample

from .cvae_model import CVAE

SAVE_DIR      = Path(__file__).parent / "weights"
BETA_WARMUP   = 20   # epochs over which beta ramps from 0 to CVAE_BETA



def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Weighted sampler for imbalanced decades 

def make_weighted_sampler(conditions: np.ndarray) -> WeightedRandomSampler:
    decade_indices = conditions[:, 3]
    counts = np.bincount(decade_indices, minlength=41).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = (1.0 / counts)[decade_indices]
    weights = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# Training entry point 

def train(
    data_path:  str,
    epochs:     int   = CVAE_EPOCHS,
    batch_size: int   = BATCH_SIZE,
    hidden_dim: int   = CVAE_HIDDEN_DIM,
    latent_dim: int   = CVAE_LATENT_DIM,
    beta:       float = CVAE_BETA,
    lr:         float = LEARNING_RATE,
    seed:       int   = RANDOM_SEED,
) -> None:
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset …")
    records              = load_dataset(data_path)
    conditions, dates    = records_to_arrays(records)

    # Shuffle + 90/10 split
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(records))
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    cond_train = conditions[train_idx]
    date_train = dates[train_idx]
    cond_val   = conditions[val_idx]
    date_val   = dates[val_idx]

    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")

    # DataLoader 
    train_dataset = TensorDataset(
        torch.tensor(cond_train, dtype=torch.long),
        torch.tensor(date_train * 1.0, dtype=torch.float32),  # already [0,1]
    )
    sampler    = make_weighted_sampler(cond_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    # Model + optimiser 
    model = CVAE(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        beta=beta,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    history   = {"total_loss": [], "recon_loss": [], "kl_loss": [], "val_accuracy": []}
    best_acc  = -1.0

    # Training loop 
    for epoch in range(1, epochs + 1):
        model.train()
        # Beta warm-up: linearly ramp from 0 to target beta
        current_beta = min(beta, beta * epoch / BETA_WARMUP)

        epoch_total = epoch_recon = epoch_kl = 0.0
        num_batches = 0
        t0 = time.time()

        for cond_batch, date_batch in train_loader:
            # date_batch is already normalised [0,1] from records_to_arrays
            # We need raw values for the loss function
            # Denorm: raw = normed * ranges + offsets
            offsets = torch.tensor([1.0, 1.0, 1800.0], device=device)
            ranges  = torch.tensor([30.0, 11.0, 400.0], device=device)
            date_raw = date_batch.to(device) * ranges + offsets   # float

            cond = cond_batch.to(device)

            optimiser.zero_grad()
            total, recon, kl = model.compute_loss(date_raw, cond, beta=current_beta)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_total += total.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()
            num_batches += 1

        scheduler.step()

        avg_total = epoch_total / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl    = epoch_kl    / num_batches
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        # Validation every 5 epochs 
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_preds = []
            cond_val_t = torch.tensor(cond_val, dtype=torch.long, device=device)

            with torch.no_grad():
                for i in range(0, len(cond_val), batch_size):
                    c_batch = cond_val_t[i : i + batch_size]
                    samples = model.sample(c_batch, num_samples=NUM_SAMPLES_AT_INFERENCE)
                    samples_np = samples.cpu().numpy()   # (batch, S, 3) normalised
                    for j, samps in enumerate(samples_np):
                        best = pick_best_sample(samps, cond_val[i + j])
                        val_preds.append(best)

            val_preds_arr = np.array(val_preds, dtype=np.float32)
            results = evaluate_predictions(cond_val, val_preds_arr, verbose=False)
            history["val_accuracy"].append(results["all"])

            if results["all"] > best_acc:
                best_acc = results["all"]
                torch.save(model.state_dict(), str(SAVE_DIR / "cvae_best.pt"))

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {avg_total:.4f} (R:{avg_recon:.4f} KL:{avg_kl:.4f}) | "
                f"β={current_beta:.2f} | "
                f"Val: {results['all']*100:.1f}%  [{time.time()-t0:.1f}s]"
            )
        else:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {avg_total:.4f} (R:{avg_recon:.4f} KL:{avg_kl:.4f}) | "
                f"β={current_beta:.2f}  [{time.time()-t0:.1f}s]"
            )

    #  Save final weights + history 
    torch.save(model.state_dict(), str(SAVE_DIR / "cvae_final.pt"))
    np.save(str(SAVE_DIR / "training_history.npy"), history)
    print(f"\nBest val accuracy: {best_acc*100:.1f}%  →  saved to {SAVE_DIR}")


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "../../data/data.txt"
    train(data_path=data_file)