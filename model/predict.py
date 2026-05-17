from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from project root 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.shared.tokenizer import (
    parse_condition_line,
    format_output_line,
    IDX_TO_DAY, IDX_TO_MONTH, DATE_OFFSETS, DATE_RANGES,
    denormalise_date, decade_idx_to_token,
)
from model.shared.validator import pick_best_sample, is_valid_date
from model.shared.config import (
    CGAN_NOISE_DIM, CGAN_HIDDEN_DIM,
    CVAE_LATENT_DIM, CVAE_HIDDEN_DIM,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS,
    TRANS_D_MODEL, TRANS_NUM_HEADS, TRANS_FF_DIM,
    TRANS_NUM_LAYERS, TRANS_LATENT_DIM, TRANS_DROPOUT,
    NUM_SAMPLES_AT_INFERENCE,
)



def load_conditions(filepath: str) -> list[np.ndarray]:
    """Return list of (4,) int64 arrays, one per non-blank input line."""
    conditions = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            arr = parse_condition_line(line)
            if arr is not None:
                conditions.append(arr)
    return conditions


# Prediction routing 

def predict_cgan(conditions: list[np.ndarray]) -> list[tuple[int, int, int]]:
    """Run inference using the Conditional GAN (TensorFlow)."""
    import tensorflow as tf
    from model.cgan.cgan_model import CGANGenerator

    weights_path = Path(__file__).parent / "cgan" / "weights" / "generator.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"CGAN weights not found at {weights_path}. "
            "Please train the model first: python -m model.cgan.cgan_trainer"
        )

    gen = CGANGenerator(hidden_dim=CGAN_HIDDEN_DIM, noise_dim=CGAN_NOISE_DIM)

    # Build model by running a dummy forward pass (required before loading weights)
    dummy_cond  = tf.zeros((1, 4), dtype=tf.int32)
    dummy_noise = tf.zeros((1, CGAN_NOISE_DIM))
    gen(dummy_cond, dummy_noise)
    gen.load_weights(str(weights_path))

    results = []
    batch_size = 256
    cond_arr = np.stack(conditions).astype(np.int32)

    for i in range(0, len(cond_arr), batch_size):
        batch_np = cond_arr[i : i + batch_size]
        batch_tf = tf.constant(batch_np, dtype=tf.int32)
        samples  = gen.generate_samples(batch_tf, num_samples=NUM_SAMPLES_AT_INFERENCE).numpy()
        for j, samps in enumerate(samples):
            best = pick_best_sample(samps, batch_np[j])
            results.append(best)
    return results


def _torch_predict(
    conditions: list[np.ndarray],
    model,
) -> list[tuple[int, int, int]]:
    """Generic inference loop for PyTorch VAE-style models (CVAE, VAETransformer)."""
    import torch
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = model.to(device)
    model.eval()

    results   = []
    cond_arr  = np.stack(conditions).astype(np.int64)
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(cond_arr), batch_size):
            batch_np = cond_arr[i : i + batch_size]
            batch_t  = torch.tensor(batch_np, dtype=torch.long, device=device)
            samples  = model.sample(batch_t, num_samples=NUM_SAMPLES_AT_INFERENCE)
            samps_np = samples.cpu().numpy()              # (batch, S, 3) normalised

            for j, samps in enumerate(samps_np):
                best = pick_best_sample(samps, batch_np[j])
                results.append(best)
    return results


def predict_cvae(conditions: list[np.ndarray]) -> list[tuple[int, int, int]]:
    import torch
    from model.cvae.cvae_model import CVAE

    weights_path = Path(__file__).parent / "cvae" / "weights" / "cvae_best.pt"
    if not weights_path.exists():
        weights_path = Path(__file__).parent / "cvae" / "weights" / "cvae_final.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"CVAE weights not found. Train first: python -m model.cvae.cvae_trainer"
        )

    model = CVAE(latent_dim=CVAE_LATENT_DIM, hidden_dim=CVAE_HIDDEN_DIM)
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    return _torch_predict(conditions, model)


def predict_lstm(conditions: list[np.ndarray]) -> list[tuple[int, int, int]]:
    import torch
    from model.lstm.lstm_model import LSTMDateGenerator, digit_sequence_to_date
    from model.shared.validator import condition_scores

    weights_path = Path(__file__).parent / "lstm" / "weights" / "lstm_best.pt"
    if not weights_path.exists():
        weights_path = Path(__file__).parent / "lstm" / "weights" / "lstm_final.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            "LSTM weights not found. Train first: python -m model.lstm.lstm_trainer"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMDateGenerator(hidden_dim=LSTM_HIDDEN_DIM, num_layers=LSTM_NUM_LAYERS)
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model  = model.to(device)
    model.eval()

    results   = []
    cond_arr  = np.stack(conditions).astype(np.int64)
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(cond_arr), batch_size):
            batch_np = cond_arr[i : i + batch_size]
            batch_t  = torch.tensor(batch_np, dtype=torch.long, device=device)
            samples  = model.generate_samples(batch_t, num_samples=NUM_SAMPLES_AT_INFERENCE)
            samps_np = samples.cpu().numpy()   # (batch, S, 8)

            for j, samp_digits in enumerate(samps_np):
                best_score = -1
                best_date  = None
                for s in range(samp_digits.shape[0]):
                    digits = samp_digits[s].tolist()
                    day, month, year = digit_sequence_to_date(digits)
                    if not is_valid_date(day, month, year):
                        continue
                    scores = condition_scores(day, month, year, batch_np[j])
                    score  = sum(scores[k] for k in ["day", "month", "leap", "decade"])
                    if score > best_score:
                        best_score = score
                        best_date  = (day, month, year)
                    if best_score == 4:
                        break
                if best_date is None:
                    best_date = digit_sequence_to_date(samp_digits[0].tolist())
                results.append(best_date)
    return results


def predict_vaetrans(conditions: list[np.ndarray]) -> list[tuple[int, int, int]]:
    import torch
    from model.vae_transformer.vae_transformer_model import VAETransformer

    weights_path = Path(__file__).parent / "vae_transformer" / "weights" / "vaetrans_best.pt"
    if not weights_path.exists():
        weights_path = Path(__file__).parent / "vae_transformer" / "weights" / "vaetrans_final.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            "VAE-Transformer weights not found. "
            "Train first: python -m model.vae_transformer.vae_transformer_trainer"
        )

    model = VAETransformer(
        d_model=TRANS_D_MODEL, n_heads=TRANS_NUM_HEADS,
        ff_dim=TRANS_FF_DIM, n_layers=TRANS_NUM_LAYERS,
        latent_dim=TRANS_LATENT_DIM, dropout=TRANS_DROPOUT,
    )
    import torch
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    return _torch_predict(conditions, model)



def write_output(
    filepath:   str,
    conditions: list[np.ndarray],
    predictions: list[tuple[int, int, int]],
) -> None:
    lines = []
    for cond, (day, month, year) in zip(conditions, predictions):
        lines.append(format_output_line(cond, day, month, year))
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Output written to: {filepath}")



MODEL_CHOICES = {
    "cgan":     predict_cgan,
    "cvae":     predict_cvae,
    "lstm":     predict_lstm,
    "vaetrans": predict_vaetrans,
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dates that satisfy given conditions."
    )
    parser.add_argument("-i", "--input",  required=True, help="Path to input conditions file")
    parser.add_argument("-o", "--output", required=True, help="Path to write predictions")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CHOICES.keys()),
        default="cvae",
        help="Which model to use for inference (default: cvae)",
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Input: {args.input}")

    # Load conditions
    conditions = load_conditions(args.input)
    if not conditions:
        print("ERROR: No valid condition lines found in input file.")
        sys.exit(1)
    print(f"Loaded {len(conditions)} condition(s).")

    # Run inference
    predict_fn   = MODEL_CHOICES[args.model]
    predictions  = predict_fn(conditions)

    # Write output
    write_output(args.output, conditions, predictions)


if __name__ == "__main__":
    main()