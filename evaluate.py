from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.shared.tokenizer import load_dataset, records_to_arrays
from model.shared.validator import evaluate_predictions, pick_best_sample


def print_comparison_table(all_results: dict[str, dict]) -> None:
    headers = ["Model", "Valid%", "Day%", "Month%", "Leap%", "Decade%", "ALL%"]
    rows = []
    for model_name, res in all_results.items():
        row = [
            model_name,
            f"{res.get('valid',0)*100:.1f}",
            f"{res.get('day',0)*100:.1f}",
            f"{res.get('month',0)*100:.1f}",
            f"{res.get('leap',0)*100:.1f}",
            f"{res.get('decade',0)*100:.1f}",
            f"{res.get('all',0)*100:.1f}",
        ]
        rows.append(row)

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "─" * (sum(col_widths) + 3 * len(col_widths))

    print("\n" + sep)
    print("  ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print(sep)
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, col_widths)))
    print(sep + "\n")


def plot_loss_curves() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping loss plots.")
        return

    model_dirs = {
        "CGAN":        Path("model/cgan/weights/training_history.npy"),
        "CVAE":   Path("model/cvae/weights/training_history.npy"),
        "LSTM":   Path("model/lstm/weights/training_history.npy"),
        "VAE-Trans": Path("model/vae_transformer/weights/training_history.npy"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History – All Models", fontsize=14)
    axes = axes.flatten()

    for ax, (name, path) in zip(axes, model_dirs.items()):
        if not path.exists():
            ax.set_title(f"{name}\n(no weights found)")
            ax.axis("off")
            continue
        hist = np.load(str(path), allow_pickle=True).item()
        ax.set_title(name)

        if "disc_loss" in hist:
            # GAN: two losses
            ax.plot(hist["disc_loss"], label="Discriminator loss")
            ax.plot(hist["gen_loss"],  label="Generator loss")
        elif "total_loss" in hist:
            ax.plot(hist["total_loss"], label="Total loss")
            ax.plot(hist["recon_loss"], label="Reconstruction loss")
            ax.plot(hist["kl_loss"],    label="KL loss")
        elif "train_loss" in hist:
            ax.plot(hist["train_loss"], label="Train CE loss")

        if "val_accuracy" in hist and hist["val_accuracy"]:
            ax2 = ax.twinx()
            # val_accuracy logged every 5 epochs
            val_epochs = list(range(0, len(hist["val_accuracy"]) * 5, 5))
            ax2.plot(val_epochs, [v * 100 for v in hist["val_accuracy"]],
                     "r--", label="Val accuracy %")
            ax2.set_ylabel("All-conditions accuracy (%)", color="r")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = "training_curves.png"
    plt.savefig(out_path, dpi=120)
    print(f"Loss curves saved to: {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare all models.")
    parser.add_argument("--data",  default="data/data.txt",
                        help="Path to data.txt (used for a held-out test split)")
    parser.add_argument("--input", default="data/example_input.txt",
                        help="Path to example_input.txt")
    parser.add_argument("--plot",  action="store_true",
                        help="Also plot training loss curves")
    args = parser.parse_args()

    # Build a small test set from data.txt
    from model.shared.tokenizer import load_dataset, records_to_arrays
    records            = load_dataset(args.data)
    conditions, dates  = records_to_arrays(records)
    rng = np.random.default_rng(42)
    test_idx = rng.choice(len(records), size=min(2000, len(records)), replace=False)
    cond_test = conditions[test_idx]
    date_test = dates[test_idx]

    print("=" * 60)
    print("Evaluating all models on a random test split")
    print("=" * 60)

    all_results: dict[str, dict] = {}

    # CGAN 
    try:
        import tensorflow as tf
        from model.cgan.cgan_model import CGANGenerator
        from model.shared.config import CGAN_HIDDEN_DIM, CGAN_NOISE_DIM, NUM_SAMPLES_AT_INFERENCE

        wp = Path("model/cgan/weights/generator.weights.h5")
        if wp.exists():
            gen = CGANGenerator(hidden_dim=CGAN_HIDDEN_DIM, noise_dim=CGAN_NOISE_DIM)
            dummy_c = tf.zeros((1, 4), dtype=tf.int32)
            dummy_n = tf.zeros((1, CGAN_NOISE_DIM))
            gen(dummy_c, dummy_n)
            gen.load_weights(str(wp))

            preds = []
            bs    = 256
            for i in range(0, len(cond_test), bs):
                cb      = tf.constant(cond_test[i:i+bs], dtype=tf.int32)
                samps   = gen.generate_samples(cb, num_samples=NUM_SAMPLES_AT_INFERENCE).numpy()
                for j, s in enumerate(samps):
                    preds.append(pick_best_sample(s, cond_test[i+j]))
            print("\n── CGAN ──")
            all_results["CGAN"] = evaluate_predictions(
                cond_test, np.array(preds, dtype=np.float32)
            )
    except Exception as e:
        print(f"CGAN evaluation skipped: {e}")

    #  CVAE 
    try:
        import torch
        from model.cvae.cvae_model import CVAE
        from model.shared.config import CVAE_LATENT_DIM, CVAE_HIDDEN_DIM

        wp = next(
            (p for p in [
                Path("model/cvae/weights/cvae_best.pt"),
                Path("model/cvae/weights/cvae_final.pt"),
            ] if p.exists()), None
        )
        if wp:
            model = CVAE(latent_dim=CVAE_LATENT_DIM, hidden_dim=CVAE_HIDDEN_DIM)
            model.load_state_dict(torch.load(str(wp), map_location="cpu"))
            model.eval()

            preds = []
            bs    = 256
            with torch.no_grad():
                for i in range(0, len(cond_test), bs):
                    cb    = torch.tensor(cond_test[i:i+bs], dtype=torch.long)
                    samps = model.sample(cb, num_samples=NUM_SAMPLES_AT_INFERENCE).numpy()
                    for j, s in enumerate(samps):
                        preds.append(pick_best_sample(s, cond_test[i+j]))
            print("\n── CVAE ──")
            all_results["CVAE"] = evaluate_predictions(
                cond_test, np.array(preds, dtype=np.float32)
            )
    except Exception as e:
        print(f"CVAE evaluation skipped: {e}")

    #  LSTM 
    try:
        import torch
        from model.lstm.lstm_model import LSTMDateGenerator, digit_sequence_to_date
        from model.shared.validator import condition_scores
        from model.shared.config import LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, NUM_SAMPLES_AT_INFERENCE

        wp = next(
            (p for p in [
                Path("model/lstm/weights/lstm_best.pt"),
                Path("model/lstm/weights/lstm_final.pt"),
            ] if p.exists()), None
        )
        if wp:
            model = LSTMDateGenerator(hidden_dim=LSTM_HIDDEN_DIM, num_layers=LSTM_NUM_LAYERS)
            model.load_state_dict(torch.load(str(wp), map_location="cpu"))
            model.eval()

            preds = []
            bs    = 256
            with torch.no_grad():
                for i in range(0, len(cond_test), bs):
                    cb    = torch.tensor(cond_test[i:i+bs], dtype=torch.long)
                    samps = model.generate_samples(cb, num_samples=NUM_SAMPLES_AT_INFERENCE).numpy()
                    for j, digit_samps in enumerate(samps):
                        best_score = -1; best_date = None
                        for s in range(digit_samps.shape[0]):
                            d, m, y = digit_sequence_to_date(digit_samps[s].tolist())
                            from model.shared.validator import is_valid_date
                            if not is_valid_date(d, m, y): continue
                            sc = condition_scores(d, m, y, cond_test[i+j])
                            score = sum(sc[k] for k in ["day","month","leap","decade"])
                            if score > best_score:
                                best_score = score; best_date = (d, m, y)
                        if best_date is None:
                            best_date = digit_sequence_to_date(digit_samps[0].tolist())
                        preds.append(best_date)
            print("\n── LSTM ──")
            all_results["LSTM"] = evaluate_predictions(
                cond_test, np.array(preds, dtype=np.float32)
            )
    except Exception as e:
        print(f"LSTM evaluation skipped: {e}")

    #  VAE-Transformer 
    try:
        import torch
        from model.vae_transformer.vae_transformer_model import VAETransformer
        from model.shared.config import (
            TRANS_D_MODEL, TRANS_NUM_HEADS, TRANS_FF_DIM,
            TRANS_NUM_LAYERS, TRANS_LATENT_DIM, TRANS_DROPOUT, NUM_SAMPLES_AT_INFERENCE
        )

        wp = next(
            (p for p in [
                Path("model/vae_transformer/weights/vaetrans_best.pt"),
                Path("model/vae_transformer/weights/vaetrans_final.pt"),
            ] if p.exists()), None
        )
        if wp:
            model = VAETransformer(
                d_model=TRANS_D_MODEL, n_heads=TRANS_NUM_HEADS,
                ff_dim=TRANS_FF_DIM, n_layers=TRANS_NUM_LAYERS,
                latent_dim=TRANS_LATENT_DIM, dropout=0.0,
            )
            model.load_state_dict(torch.load(str(wp), map_location="cpu"))
            model.eval()

            preds = []
            bs    = 256
            with torch.no_grad():
                for i in range(0, len(cond_test), bs):
                    cb    = torch.tensor(cond_test[i:i+bs], dtype=torch.long)
                    samps = model.sample(cb, num_samples=NUM_SAMPLES_AT_INFERENCE).numpy()
                    for j, s in enumerate(samps):
                        preds.append(pick_best_sample(s, cond_test[i+j]))
            print("\n── VAE-Transformer ──")
            all_results["VAE-Trans"] = evaluate_predictions(
                cond_test, np.array(preds, dtype=np.float32)
            )
    except Exception as e:
        print(f"VAE-Transformer evaluation skipped: {e}")

    # Summary table 
    if all_results:
        print_comparison_table(all_results)
    
        with open("evaluation_results.txt", "w") as f:
            headers = ["Model", "Valid%", "Day%", "Month%", "Leap%", "Decade%", "ALL%"]

            f.write(" | ".join(headers) + "\n")
            f.write("-" * 80 + "\n")

            for model_name, res in all_results.items():
                row = [
                    model_name,
                    f"{res.get('valid',0)*100:.1f}",
                    f"{res.get('day',0)*100:.1f}",
                    f"{res.get('month',0)*100:.1f}",
                    f"{res.get('leap',0)*100:.1f}",
                    f"{res.get('decade',0)*100:.1f}",
                    f"{res.get('all',0)*100:.1f}",
                ]
                f.write(" | ".join(row) + "\n")
        

        print("Evaluation results saved to evaluation_results.txt")

     

    if args.plot:
        plot_loss_curves()


if __name__ == "__main__":
    main()