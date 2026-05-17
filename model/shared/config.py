"""
config.py
---------
Central configuration file for the Date Generator project.
All hyperparameters and constants live here so you only need
to change one file when experimenting.
"""

# ── Date range ────────────────────────────────────────────────────────────────
MIN_YEAR = 1800
MAX_YEAR = 2200

# ── Vocabulary sizes ──────────────────────────────────────────────────────────
NUM_DAYS_OF_WEEK = 7      # MON … SUN
NUM_MONTHS       = 12     # JAN … DEC
NUM_LEAP_OPTIONS = 2      # True / False
NUM_DECADES      = 41     # 1800–2200 → 40 full decades + 1

# ── Derived sizes ─────────────────────────────────────────────────────────────
# Condition embedding total dim = 4 * EMBEDDING_DIM (one per condition field)
EMBEDDING_DIM  = 16
CONDITION_DIM  = 4 * EMBEDDING_DIM   # 64

# Date is represented as (day, month, year) → 3 raw numbers
DATE_DIM = 3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 256
LEARNING_RATE = 5e-5
RANDOM_SEED   = 42

# ── CGAN specifics ────────────────────────────────────────────────────────────
CGAN_NOISE_DIM  = 32
CGAN_HIDDEN_DIM = 256
CGAN_EPOCHS     = 40

# ── CVAE specifics ────────────────────────────────────────────────────────────
CVAE_LATENT_DIM = 32
CVAE_HIDDEN_DIM = 128
CVAE_BETA       = 1.0    # KL weight (increase for disentanglement)
CVAE_EPOCHS     = 40

# ── LSTM specifics ────────────────────────────────────────────────────────────
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS     = 40
LSTM_DROPOUT    = 0.2

# ── VAE-Transformer specifics ─────────────────────────────────────────────────
TRANS_LATENT_DIM  = 64
TRANS_D_MODEL     = 128
TRANS_NUM_HEADS   = 4
TRANS_FF_DIM      = 256
TRANS_NUM_LAYERS  = 2
TRANS_DROPOUT     = 0.1
TRANS_BETA        = 1.0
TRANS_EPOCHS      = 40

# ── Evaluation ────────────────────────────────────────────────────────────────
# How many candidate dates to generate per condition set before picking the best
NUM_SAMPLES_AT_INFERENCE = 50