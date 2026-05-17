# Date Generator — Four Generative Models

> **Assignment 1 · Deep Generative Models**
> Generates a valid calendar date from a set of four conditions
> (day-of-week, month, leap year, decade) using four different neural network architectures.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [The Four Models](#the-four-models)
4. [Architecture Changes and Design Decisions](#architecture-changes-and-design-decisions)
5. [Bonus Features (per model)](#bonus-features-per-model)
6. [How to Train](#how-to-train)
7. [How to Run Inference](#how-to-run-inference)
8. [How to Evaluate](#how-to-evaluate)
9. [Condition Format Reference](#condition-format-reference)

---

## Project Structure

```
date_generator/
├── data/
│   ├── data.txt                   ← full dataset (146 461 lines)
│   └── example_input.txt          ← conditions only (1 465 lines)
│
├── model/
│   ├── predict.py                 ← MAIN inference entry point (required by spec)
│   │
│   ├── shared/
│   │   ├── config.py              ← all hyperparameters in one place
│   │   ├── tokenizer.py           ← parsing, normalisation, vocabulary
│   │   └── validator.py           ← date validity checks, condition accuracy
│   │
│   ├── cgan/
│   │   ├── cgan_model.py          ← Generator + Discriminator (TensorFlow)
│   │   ├── cgan_trainer.py        ← WGAN-GP training loop (TensorFlow)
│   │   └── weights/               ← saved after training
│   │
│   ├── cvae/
│   │   ├── cvae_model.py          ← CVAE encoder/decoder (PyTorch)
│   │   ├── cvae_trainer.py        ← training loop with beta warm-up (PyTorch)
│   │   └── weights/
│   │
│   ├── lstm/
│   │   ├── lstm_model.py          ← autoregressive LSTM (PyTorch)
│   │   ├── lstm_trainer.py        ← training loop (PyTorch)
│   │   └── weights/
│   │
│   └── vae_transformer/
│       ├── vae_transformer_model.py ← VAE with Transformer enc/dec (PyTorch)
│       ├── vae_transformer_trainer.py
│       └── weights/
│
├── evaluate.py                    ← compare all models, plot loss curves
├── environment.yml                ← conda environment spec
└── README.md
```

---

## Quick Start

### 1. Create the conda environment
```bash
conda env create -f environment.yml
conda activate date_generator
```

### 2. Place your data files
```
date_generator/data/data.txt
date_generator/data/example_input.txt
```

### 3. Train all models (run from `date_generator/`)
```bash
# Model 1 – Conditional GAN (TensorFlow)
python -m model.cgan.cgan_trainer data/data.txt

# Model 2 – Conditional VAE (PyTorch)
python -m model.cvae.cvae_trainer data/data.txt

# Model 3 – LSTM (PyTorch)
python -m model.lstm.lstm_trainer data/data.txt

# Model 4 – VAE-Transformer (PyTorch)
python -m model.vae_transformer.vae_transformer_trainer data/data.txt
```

### 4. Run inference (required command from the spec)
```bash
python model/predict.py -i data/example_input.txt -o outputs/predictions.txt --model cvae
```

---

## The Four Models

| # | Model | Framework | Category |
|---|-------|-----------|----------|
| 1 | **Conditional GAN (CGAN)** | TensorFlow | *From the course* |
| 2 | **Conditional VAE (CVAE)** | PyTorch | *From the course* |
| 3 | **Autoregressive LSTM** | PyTorch | *Outside the course* |
| 4 | **VAE-Transformer** | PyTorch | *Outside the course* |

---

## Architecture Changes and Design Decisions

### Problem Formulation
The date is represented as three continuous numbers: **(day, month, year)**.  
All three are normalised to **[0, 1]** before being fed to the network:
```
day_norm   = (day − 1)   / 30
month_norm = (month − 1) / 11
year_norm  = (year − 1800) / 400
```
This ensures day, month, and year contribute **equally** to the loss, despite having different absolute scales.

### Condition Encoding
Instead of raw integer indices, each of the 4 condition fields is passed through a **learned embedding layer** (dimension 16).  
The four embeddings are concatenated → a 64-dimensional condition vector.  
**Why?** Integers like `6` and `7` are numerically close but semantically very different (e.g. decade 180 vs 181). Embeddings let the network learn the right similarity structure.

### Model 1 – CGAN (TensorFlow)
- **Generator**: condition embedding + Gaussian noise → 3-float date via MLP with LeakyReLU + BatchNorm.
- **Discriminator**: condition embedding + date vector → real/fake score via MLP with LeakyReLU + Dropout.
- **Changes from reference**: Added embedding layers to both Generator and Discriminator (reference passed raw integers). Switched from vanilla GAN loss to **WGAN-GP** (Wasserstein + Gradient Penalty) which is far more stable for structured data.

### Model 2 – CVAE (PyTorch)
- **Encoder** `q(z|x,c)`: normalised date + condition embedding → (mu, log_var) in latent space.
- **Decoder** `p(x|z,c)`: sampled z + condition embedding → reconstructed normalised date.
- **Changes from reference**: Hidden layers widened (128→256, 64→128). LayerNorm added after every hidden block. Shared `ConditionEmbedder` module (reference duplicated it). Beta-KL warm-up implemented in trainer (not present in reference).

### Model 3 – LSTM (autoregressive, PyTorch)
Treats date generation as a **sequence problem**: the date `dd-mm-yyyy` is produced digit-by-digit.

**Why digit-by-digit?**  
The LSTM knows what digits it has already produced, so it can learn constraints like "if the month is 02 (February), day must not exceed 28 or 29". This is a form of self-conditioning absent in the VAE models.

- Input: sequence of digit tokens `[SOS, d₀, d₁, …, d₆]` (teacher-forcing during training).
- Output: one-hot logit per digit position (vocabulary = 0-9, SOS, EOS).
- Condition → **LSTM initial hidden state** (h₀, c₀) via a linear projection.
- Loss: label-smoothed cross-entropy (smoothing=0.1).

### Model 4 – VAE-Transformer (PyTorch)
Combines the VAE framework with Transformer self-attention.

**Why Transformer?**  
Self-attention lets the encoder relate all 4 condition fields to each other simultaneously. For example, it can learn that `[FEB, True]` (February + leap year) allows day 29, while `[FEB, False]` does not.

- **Encoder**: 4 condition tokens + 1 date token → self-attention → pool → (mu, log_var).
- **Decoder**: z token cross-attends over condition tokens → date.
- Uses **Pre-LN** (layer norm before attention, not after) for more stable training.
- Uses **OneCycleLR** schedule and **mixed-precision** training (when GPU is available).

---

## Bonus Features (per model)

All models share:
-  Manual random seed (`RANDOM_SEED = 42`) for reproducible results.
-  90/10 train/validation split, shuffled before split.
-  **Weighted random sampler** by decade to address class imbalance.
-  **Condition-accuracy** (not loss) tracked on the validation set during training — this is the real metric since many valid dates exist per condition.
- Best checkpoint saved (by validation all-conditions accuracy).

Model-specific bonuses:
| Model | Extra bonus |
|-------|------------|
| CGAN | WGAN-GP gradient penalty; discriminator trained 5× per generator step |
| CVAE | Beta-KL warm-up schedule (avoids posterior collapse); cosine LR |
| LSTM | Label smoothing; gradient clipping; temperature-controlled sampling |
| VAE-Transformer | Mixed-precision (AMP); OneCycleLR; Pre-LN architecture; cross-attention decoder |

---

## How to Train

All trainers accept a path to `data.txt` as their first argument.
Run from the `date_generator/` directory.

```bash
# Model 1 – CGAN
python -m model.cgan.cgan_trainer data/data.txt

# Model 2 – CVAE
python -m model.cvae.cvae_trainer data/data.txt

# Model 3 – LSTM
python -m model.lstm.lstm_trainer data/data.txt

# Model 4 – VAE-Transformer
python -m model.vae_transformer.vae_transformer_trainer data/data.txt
```

Weights are saved to `model/<name>/weights/` automatically.

---

## How to Run Inference

The **required** command from the assignment spec:

```bash
python model/predict.py -i data/example_input.txt -o outputs/predictions.txt
```

Options:
```
-i   path to input conditions file
-o   path to write predictions
--model  cgan | cvae | lstm | vaetrans   (default: cvae)
```

The output file matches `data.txt` format exactly:
```
[WED] [JAN] [False] [180] 1-1-1800
[MON] [JAN] [False] [190] 3-1-1901
...
```

Inference uses **best-of-50 sampling**: 50 candidate dates are drawn for each condition set, and the one satisfying the most sub-conditions is selected.

---

## How to Evaluate

```bash
# Evaluate all available trained models and print a table
python evaluate.py --data data/data.txt --input data/example_input.txt

# Also plot training loss curves
python evaluate.py --data data/data.txt --input data/example_input.txt --plot
```

The table shows per-condition accuracy:

| Column | What it measures |
|--------|-----------------|
| `Valid%` | Generated date is a real calendar date |
| `Day%` | Day-of-week matches condition |
| `Month%` | Month matches condition |
| `Leap%` | Leap year status matches condition |
| `Decade%` | Year is in the correct decade |
| `ALL%` | All four conditions satisfied simultaneously |

---

## Condition Format Reference

```
[DAY] [MONTH] [LEAP] [DECADE]

DAY    : MON TUE WED THU FRI SAT SUN
MONTH  : JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC
LEAP   : True | False
DECADE : 180 = 1800-1809 … 220 = 2200-2209  (step of 1 per decade)
```

Example input line:
```
[WED] [JAN] [False] [196]
```
Expected output: any date that is a **Wednesday** in **January** of a **non-leap** year in the **1960s**.

---

## Hyperparameter Reference

All hyperparameters are in `model/shared/config.py`. Change values there to experiment without touching the model or trainer code.

```python
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-3
RANDOM_SEED    = 42

CGAN_EPOCHS    = 80
CVAE_EPOCHS    = 80
LSTM_EPOCHS    = 60
TRANS_EPOCHS   = 80
```
