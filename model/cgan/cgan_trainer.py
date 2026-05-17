from __future__ import annotations

import os
import time
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..shared.config import (
    BATCH_SIZE, LEARNING_RATE, RANDOM_SEED,
    CGAN_NOISE_DIM, CGAN_HIDDEN_DIM, CGAN_EPOCHS,
)
from ..shared.tokenizer import load_dataset, records_to_arrays
from ..shared.validator import evaluate_predictions, denormalise_date, pick_best_sample
from .cgan_model import CGANGenerator, CGANDiscriminator

SAVE_DIR    = Path(__file__).parent / "weights"
DISC_STEPS  = 3      # discriminator updates per generator update
GP_LAMBDA   = 10.0   # gradient penalty weight


# for reproducibility 
def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


#  WGAN-GP gradient penalty 

def gradient_penalty(
    discriminator: CGANDiscriminator,
    conditions:    tf.Tensor,
    real_dates:    tf.Tensor,
    fake_dates:    tf.Tensor,
) -> tf.Tensor:
    # Compute the WGAN-GP gradient penalty term
    batch = tf.shape(real_dates)[0]
    alpha = tf.random.uniform((batch, 1), 0.0, 1.0)
    interpolated = real_dates + alpha * (fake_dates - real_dates)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        score = discriminator(conditions, interpolated, training=True)

    grads    = tape.gradient(score, interpolated)
    norm = tf.norm(grads, axis=-1)
    penalty  = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty


# Loss functions 

def discriminator_loss(real_logits: tf.Tensor, fake_logits: tf.Tensor) -> tf.Tensor:
    # WGAN critic loss (maximise real, minimise fake)
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)


def generator_loss(fake_logits: tf.Tensor) -> tf.Tensor:
    # WGAN generator loss 
    return -tf.reduce_mean(fake_logits)


# One-step train functions 

@tf.function
def train_discriminator_step(
    generator:     CGANGenerator,
    discriminator: CGANDiscriminator,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    conditions:    tf.Tensor,
    real_dates:    tf.Tensor,
) -> tf.Tensor:
    batch    = tf.shape(conditions)[0]
    noise    = tf.random.normal((batch, CGAN_NOISE_DIM))

    with tf.GradientTape() as tape:
        fake_dates  = generator(conditions, noise, training=False)
        real_logits = discriminator(conditions, real_dates,  training=True)
        fake_logits = discriminator(conditions, fake_dates,  training=True)
        loss_d      = discriminator_loss(real_logits, fake_logits)
        gp          = gradient_penalty(discriminator, conditions, real_dates, fake_dates)
        total_d     = loss_d + GP_LAMBDA * gp

    grads = tape.gradient(total_d, discriminator.trainable_variables)
    grads = [tf.clip_by_norm(g, 5.0) for g in grads]

    disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return total_d


@tf.function
def train_generator_step(
    generator:      CGANGenerator,
    discriminator:  CGANDiscriminator,
    gen_optimizer:  tf.keras.optimizers.Optimizer,
    conditions:     tf.Tensor,
) -> tf.Tensor:
    batch = tf.shape(conditions)[0]
    noise = tf.random.normal((batch, CGAN_NOISE_DIM))

    with tf.GradientTape() as tape:
        fake_dates  = generator(conditions, noise, training=True)
        fake_logits = discriminator(conditions, fake_dates, training=False)
        loss_g      = generator_loss(fake_logits)

    grads = tape.gradient(loss_g, generator.trainable_variables)
    grads = [tf.clip_by_norm(g, 5.0) for g in grads]

    gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return loss_g


# Main training entry point 

def train(
    data_path:   str,
    epochs:      int  = CGAN_EPOCHS,
    batch_size:  int  = BATCH_SIZE,
    hidden_dim:  int  = CGAN_HIDDEN_DIM,
    noise_dim:   int  = CGAN_NOISE_DIM,
    lr:          float = LEARNING_RATE,
    seed:        int  = RANDOM_SEED,
) -> None:
    set_seeds(seed)

    # Load data 
    print("Loading dataset …")
    records           = load_dataset(data_path)
    conditions, dates = records_to_arrays(records)

    # Shuffle + train/val split (90/10)
    idx = np.random.permutation(len(records))
    split  = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    cond_train, date_train = conditions[train_idx], dates[train_idx]
    cond_val,   date_val   = conditions[val_idx],   dates[val_idx]
    print(f"  Train samples: {len(train_idx):,}  |  Val samples: {len(val_idx):,}")

    # bonus: class-weighted sampling by decade 
    decade_counts = np.bincount(cond_train[:, 3], minlength=41).astype(np.float32)
    decade_counts = np.where(decade_counts == 0, 1, decade_counts)
    sample_weights = (1.0 / decade_counts)[cond_train[:, 3]]
    sample_weights /= sample_weights.sum()

    # Build tf.data pipeline
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(cond_train, tf.int32), tf.cast(date_train, tf.float32))
    )
    dataset = (
        dataset
        .shuffle(buffer_size=10_000, seed=seed)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build models 
    generator     = CGANGenerator    (hidden_dim=hidden_dim, noise_dim=noise_dim)
    discriminator = CGANDiscriminator(hidden_dim=hidden_dim)

    gen_optimizer  = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)
    disc_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    history = {"disc_loss": [], "gen_loss": [], "val_accuracy": []}

    # Training loop 
    for epoch in range(1, epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches  = 0
        t0 = time.time()

        for cond_batch, date_batch in dataset:
            # Train discriminator DISC_STEPS times per generator step
            for _ in range(DISC_STEPS):
                d_loss = train_discriminator_step(
                    generator, discriminator, disc_optimizer,
                    cond_batch, date_batch,
                )
            g_loss = train_generator_step(
                generator, discriminator, gen_optimizer,
                cond_batch,
            )
            epoch_d_loss += float(d_loss)
            epoch_g_loss += float(g_loss)
            num_batches  += 1

        avg_d = epoch_d_loss / num_batches
        avg_g = epoch_g_loss / num_batches
        history["disc_loss"].append(avg_d)
        history["gen_loss"].append(avg_g)

        # Validation: condition accuracy 
        if epoch % 10 == 0 or epoch == 1:
            val_preds = []
            val_cond_tf = tf.cast(cond_val, tf.int32)
            for i in range(0, len(cond_val), batch_size):
                c_batch   = val_cond_tf[i : i + batch_size]
                # sample best of 20
                samples   = generator.generate_samples(c_batch, num_samples=20).numpy()
                for j, samps in enumerate(samples):
                    best = pick_best_sample(samps, cond_val[i + j])
                    val_preds.append(best)

            val_preds_arr = np.array(val_preds, dtype=np.float32)   # (N, 3)
            results = evaluate_predictions(cond_val, val_preds_arr, verbose=False)
            history["val_accuracy"].append(results["all"])
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"D-loss: {avg_d:+.4f} | G-loss: {avg_g:+.4f} | "
                f"Val all-conditions: {results['all']*100:.1f}%  "
                f"[{time.time()-t0:.1f}s]"
            )
        else:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"D-loss: {avg_d:+.4f} | G-loss: {avg_g:+.4f}  "
                f"[{time.time()-t0:.1f}s]"
            )

    # Save weights 
    generator    .save_weights(str(SAVE_DIR / "generator.weights.h5"))
    discriminator.save_weights(str(SAVE_DIR / "discriminator.weights.h5"))
    np.save(str(SAVE_DIR / "training_history.npy"), history)
    print(f"\nWeights saved to {SAVE_DIR}")


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "../../data/data.txt"
    train(data_path=data_file)