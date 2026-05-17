from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.config import (
    NUM_DAYS_OF_WEEK, NUM_MONTHS, NUM_LEAP_OPTIONS, NUM_DECADES,
    EMBEDDING_DIM, CONDITION_DIM, DATE_DIM,
    TRANS_LATENT_DIM, TRANS_D_MODEL, TRANS_NUM_HEADS,
    TRANS_FF_DIM, TRANS_NUM_LAYERS, TRANS_DROPOUT,
)

DATE_OFFSETS = torch.tensor([1.0, 1.0, 1800.0])
DATE_RANGES  = torch.tensor([30.0, 11.0, 400.0])



class ConditionTokenEmbedder(nn.Module):
   
    def __init__(self, d_model: int = TRANS_D_MODEL) -> None:
        super().__init__()
        self.day_embed    = nn.Embedding(NUM_DAYS_OF_WEEK, d_model)
        self.month_embed  = nn.Embedding(NUM_MONTHS,       d_model)
        self.leap_embed   = nn.Embedding(NUM_LEAP_OPTIONS, d_model)
        self.decade_embed = nn.Embedding(NUM_DECADES,      d_model)
        # One learnable position per condition slot
        self.pos_embed    = nn.Embedding(4, d_model)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(4, device=conditions.device)   # [0, 1, 2, 3]
        d   = self.day_embed   (conditions[:, 0]).unsqueeze(1)
        m   = self.month_embed (conditions[:, 1]).unsqueeze(1)
        l   = self.leap_embed  (conditions[:, 2]).unsqueeze(1)
        e   = self.decade_embed(conditions[:, 3]).unsqueeze(1)
        tokens = torch.cat([d, m, l, e], dim=1)          # (batch, 4, d_model)
        tokens = tokens + self.pos_embed(pos).unsqueeze(0)
        return tokens


# Pre-LN Transformer encoder block 

class TransformerEncoderBlock(nn.Module):
    # single Transformer encoder layer with Pre-LN for stable training

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        ff_dim:   int,
        dropout:  float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention (Pre-LN)
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x    = x + y
        # Feed-forward (Pre-LN)
        x    = x + self.ff(self.norm2(x))
        return x


# Pre-LN Transformer decoder block 

class TransformerDecoderBlock(nn.Module):
    # Cross-attention block: query = latent z tokens, key/value = condition tokens.

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim:  int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        self.self_attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff          = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query:   torch.Tensor,   # (batch, q_len, d_model)  – latent tokens
        context: torch.Tensor,   # (batch, 4, d_model)      – condition tokens
    ) -> torch.Tensor:
        # Self-attention on query
        y, _ = self.self_attn(self.norm1(query), self.norm1(query), self.norm1(query))
        query = query + y
        # Cross-attention: query attends to conditions
        y, _ = self.cross_attn(self.norm2(query), context, context)
        query = query + y
        # Feed-forward
        query = query + self.ff(self.norm3(query))
        return query


#  Full VAE-Transformer 

class VAETransformer(nn.Module):
    # Variational Autoencoder with Transformer encoder/decoder.

    # Encoder: condition tokens + date token to (mu, log_var) in latent space.
    # Decoder: z + condition tokens → normalised date in [0, 1]³.

    def __init__(
        self,
        d_model:    int   = TRANS_D_MODEL,
        n_heads:    int   = TRANS_NUM_HEADS,
        ff_dim:     int   = TRANS_FF_DIM,
        n_layers:   int   = TRANS_NUM_LAYERS,
        latent_dim: int   = TRANS_LATENT_DIM,
        dropout:    float = TRANS_DROPOUT,
        beta:       float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta

        # Encoder 
        self.cond_token_embedder = ConditionTokenEmbedder(d_model)

        # Project date (3 floats normalised) into d_model space
        self.date_projection = nn.Linear(DATE_DIM, d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        # Pool: take mean over sequence, then project to mu and log_var
        self.enc_norm  = nn.LayerNorm(d_model)
        self.mu_head   = nn.Linear(d_model, latent_dim)
        self.lv_head   = nn.Linear(d_model, latent_dim)

        # Decoder 
        # Project z into d_model space (treated as a single query token)
        self.z_projection = nn.Linear(latent_dim, d_model)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.dec_norm     = nn.LayerNorm(d_model)
        self.output_head  = nn.Sequential(
            nn.Linear(d_model, DATE_DIM),
            nn.Sigmoid(),
        )

    # Encoding 

    def encode(
        self,
        date_norm:  torch.Tensor,    # (batch, 3) in [0,1]
        conditions: torch.Tensor,    # (batch, 4) int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var) each (batch, latent_dim)."""
        cond_tokens = self.cond_token_embedder(conditions)     # (batch, 4, d_model)
        date_token  = self.date_projection(date_norm).unsqueeze(1)  # (batch, 1, d_model)
        x = torch.cat([cond_tokens, date_token], dim=1)        # (batch, 5, d_model)

        for layer in self.encoder_layers:
            x = layer(x)
        x = self.enc_norm(x)

        # Mean pooling over all tokens
        pooled  = x.mean(dim=1)                                # (batch, d_model)
        mu      = self.mu_head(pooled)
        log_var = self.lv_head(pooled)
        return mu, log_var

    # Reparameterisation 

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decoding 

    def decode(
        self,
        z:          torch.Tensor,   # (batch, latent_dim)
        conditions: torch.Tensor,   # (batch, 4)
    ) -> torch.Tensor:
        """Returns (batch, 3) in [0,1]."""
        cond_tokens = self.cond_token_embedder(conditions)       # (batch, 4, d_model)
        query       = self.z_projection(z).unsqueeze(1)          # (batch, 1, d_model)

        for layer in self.decoder_layers:
            query = layer(query, cond_tokens)
        query = self.dec_norm(query)

        return self.output_head(query.squeeze(1))                # (batch, 3)

    # Forward 

    def forward(
        self,
        date_raw:   torch.Tensor,
        conditions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device  = date_raw.device
        offsets = DATE_OFFSETS.to(device)
        ranges  = DATE_RANGES .to(device)
        date_norm = (date_raw.float() - offsets) / ranges       # (batch, 3) in [0,1]

        mu, log_var = self.encode(date_norm, conditions)
        z           = self.reparameterise(mu, log_var)
        recon_norm  = self.decode(z, conditions)

        return recon_norm, mu, log_var, date_norm

    # Loss 

    def compute_loss(
        self,
        date_raw:   torch.Tensor,
        conditions: torch.Tensor,
        beta:       float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if beta is None:
            beta = self.beta
        recon_norm, mu, log_var, date_norm = self.forward(date_raw, conditions)
        recon_loss = F.mse_loss(recon_norm, date_norm)
        kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total      = recon_loss + beta * kl_loss
        return total, recon_loss, kl_loss

    # Inference 

    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,   # (batch, 4)
        num_samples: int = 1,
    ) -> torch.Tensor:
        # Returns (batch, num_samples, 3) normalised [0,1]
        device     = conditions.device
        batch_size = conditions.shape[0]

        cond_rep = conditions.repeat_interleave(num_samples, dim=0)  # (batch*S, 4)
        z        = torch.randn(batch_size * num_samples, self.latent_dim, device=device)
        pred     = self.decode(z, cond_rep)                          # (batch*S, 3)

        return pred.view(batch_size, num_samples, DATE_DIM)