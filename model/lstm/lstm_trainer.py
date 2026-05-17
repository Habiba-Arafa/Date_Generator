from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..shared.config import (
    BATCH_SIZE, LEARNING_RATE, RANDOM_SEED,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_EPOCHS,
    NUM_SAMPLES_AT_INFERENCE,
)
from ..shared.tokenizer import load_dataset, records_to_arrays
from ..shared.validator import (
    evaluate_predictions, pick_best_sample,
    is_valid_date, denormalise_date,
)
from .lstm_model import (
    LSTMDateGenerator,
    date_to_digit_sequence, digit_sequence_to_date,
    SEQ_LEN, SOS_TOKEN, EOS_TOKEN,
)

SAVE_DIR = Path(__file__).parent / "weights"



def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Weighted sampler 

def make_weighted_sampler(conditions: np.ndarray) -> WeightedRandomSampler:
    decade_indices = conditions[:, 3]
    counts  = np.bincount(decade_indices, minlength=41).astype(np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = (1.0 / counts)[decade_indices]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float),
        num_samples=len(weights),
        replacement=True,
    )


# Dataset builder 

def build_digit_sequences(
    conditions: np.ndarray,
    dates_norm: np.ndarray,
) -> torch.Tensor:
    
    # Convert normalised date arrays → digit sequence tensors.
    sequences = []
    for i in range(len(dates_norm)):
        day, month, year = denormalise_date(dates_norm[i])
        seq = date_to_digit_sequence(day, month, year)
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)   # (N, 8)


# Label-smoothed cross-entropy 


def label_smoothed_ce(
    logits:  torch.Tensor,   # (batch * seq_len, vocab_size)
    targets: torch.Tensor,   # (batch * seq_len,)
    smoothing: float = 0.1,
) -> torch.Tensor:
    n_classes = logits.size(-1)
    with torch.no_grad():
        smooth_targets = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(smooth_targets * log_probs).sum(dim=-1).mean()


# Inference helper 

def predict_best_date(
    model:       LSTMDateGenerator,
    conditions:  torch.Tensor,
    cond_np:     np.ndarray,
    num_samples: int = NUM_SAMPLES_AT_INFERENCE,
) -> list[tuple[int, int, int]]:
    
    # For each condition row, generate num_samples digit sequences
    # and return the (day, month, year) that satisfies the most conditions.
    
    batch   = conditions.shape[0]
    samples = model.generate_samples(conditions, num_samples=num_samples)
    # samples: (batch, num_samples, SEQ_LEN)

    results = []
    for i in range(batch):
        best_score = -1
        best_date  = None
        for s in range(num_samples):
            digits = samples[i, s].tolist()
            day, month, year = digit_sequence_to_date(digits)
            if not is_valid_date(day, month, year):
                continue
            # Count met conditions
            from ..shared.validator import condition_scores
            scores = condition_scores(day, month, year, cond_np[i])
            score  = sum(scores[k] for k in ["day", "month", "leap", "decade"])
            if score > best_score:
                best_score = score
                best_date  = (day, month, year)
            if best_score == 4:
                break
        if best_date is None:
            digits = samples[i, 0].tolist()
            best_date = digit_sequence_to_date(digits)
        results.append(best_date)
    return results


# Training entry point 

def train(
    data_path:   str,
    epochs:      int   = LSTM_EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    hidden_dim:  int   = LSTM_HIDDEN_DIM,
    num_layers:  int   = LSTM_NUM_LAYERS,
    lr:          float = LEARNING_RATE,
    seed:        int   = RANDOM_SEED,
) -> None:
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset …")
    records            = load_dataset(data_path)
    conditions, dates  = records_to_arrays(records)

    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(records))
    split = int(0.9 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]

    cond_train = conditions[tr_idx]
    date_train = dates[tr_idx]
    cond_val   = conditions[val_idx]
    date_val   = dates[val_idx]

    # Build digit sequences from dates
    print("Building digit sequences …")
    seq_train = build_digit_sequences(cond_train, date_train)   # (N, 8)
    seq_val   = build_digit_sequences(cond_val,   date_val)

    print(f"  Train: {len(tr_idx):,}  |  Val: {len(val_idx):,}")

    dataset = TensorDataset(
        torch.tensor(cond_train, dtype=torch.long),
        seq_train,
    )
    sampler     = make_weighted_sampler(cond_train)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0
    )

    # Model 
    model     = LSTMDateGenerator(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    history  = {"train_loss": [], "val_accuracy": []}
    best_acc = -1.0

    # Training loop 
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0
        t0 = time.time()

        for cond_batch, seq_batch in train_loader:
            cond_batch = cond_batch.to(device)
            seq_batch  = seq_batch .to(device)

            logits = model(cond_batch, seq_batch)   # (batch, SEQ_LEN, VOCAB_SIZE)

            # Reshape for loss: (batch*SEQ_LEN, VOCAB_SIZE) vs (batch*SEQ_LEN,)
            b, s, v = logits.shape
            loss = label_smoothed_ce(
                logits.reshape(b * s, v),
                seq_batch.reshape(b * s),
                smoothing=0.1,
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss  += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)

        # Validation every 5 epochs 
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_preds = []
            cond_val_t = torch.tensor(cond_val, dtype=torch.long, device=device)

            for i in range(0, len(cond_val), batch_size):
                c_batch  = cond_val_t[i : i + batch_size]
                c_np     = cond_val[i : i + batch_size]
                preds    = predict_best_date(model, c_batch, c_np, num_samples=NUM_SAMPLES_AT_INFERENCE)
                val_preds.extend(preds)

            val_arr  = np.array(val_preds, dtype=np.float32)
            results  = evaluate_predictions(cond_val, val_arr, verbose=False)
            history["val_accuracy"].append(results["all"])

            if results["all"] > best_acc:
                best_acc = results["all"]
                torch.save(model.state_dict(), str(SAVE_DIR / "lstm_best.pt"))

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val: {results['all']*100:.1f}%  [{time.time()-t0:.1f}s]"
            )
        else:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f}  [{time.time()-t0:.1f}s]")

    torch.save(model.state_dict(), str(SAVE_DIR / "lstm_final.pt"))
    np.save(str(SAVE_DIR / "training_history.npy"), history)
    print(f"\nBest val accuracy: {best_acc*100:.1f}%  →  {SAVE_DIR}")


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "../../data/data.txt"
    train(data_path=data_file)