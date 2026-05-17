from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..shared.config import (
    NUM_DAYS_OF_WEEK, NUM_MONTHS, NUM_LEAP_OPTIONS, NUM_DECADES,
    EMBEDDING_DIM, CONDITION_DIM, DATE_DIM,
    CVAE_LATENT_DIM, CVAE_HIDDEN_DIM,
)

# Normalisation constants
DATE_OFFSETS = torch.tensor([1.0, 1.0, 1800.0])
DATE_RANGES  = torch.tensor([30.0, 11.0, 400.0])


# Shared condition embedder 

class ConditionEmbedder(nn.Module):
   
    def __init__(self, embed_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.day_embed    = nn.Embedding(NUM_DAYS_OF_WEEK, embed_dim)
        self.month_embed  = nn.Embedding(NUM_MONTHS,       embed_dim)
        self.leap_embed   = nn.Embedding(NUM_LEAP_OPTIONS, embed_dim)
        self.decade_embed = nn.Embedding(NUM_DECADES,      embed_dim)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        d = self.day_embed   (conditions[:, 0])
        m = self.month_embed (conditions[:, 1])
        l = self.leap_embed  (conditions[:, 2])
        e = self.decade_embed(conditions[:, 3])
        return torch.cat([d, m, l, e], dim=1)   


# Encoder 

class CVAEEncoder(nn.Module):
    
    def __init__(
        self,
        latent_dim:  int = CVAE_LATENT_DIM,
        hidden_dim:  int = CVAE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        input_dim = DATE_DIM + CONDITION_DIM   # 3 + 64 = 67

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        date_norm: torch.Tensor, 
        cond_emb:  torch.Tensor,  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x      = torch.cat([date_norm, cond_emb], dim=1)
        hidden = self.mlp(x)
        return self.mu_head(hidden), self.log_var_head(hidden)


#  Decoder 

class CVAEDecoder(nn.Module):
  
    def __init__(
        self,
        latent_dim: int = CVAE_LATENT_DIM,
        hidden_dim: int = CVAE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + CONDITION_DIM

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, DATE_DIM),
            nn.Sigmoid(),    # keeps output in [0,1] for MSE / BCE
        )

    def forward(
        self,
        z:       torch.Tensor,   # (batch, latent_dim)
        cond_emb: torch.Tensor,  # (batch, CONDITION_DIM)
    ) -> torch.Tensor:
        x = torch.cat([z, cond_emb], dim=1)
        return self.mlp(x)      # (batch, 3) in [0,1]


# Full CVAE 

class CVAE(nn.Module):
    # Conditional VAE that generates normalised date vectors.

    # Call .sample(conditions) at inference time.
    
    def __init__(
        self,
        embed_dim:  int   = EMBEDDING_DIM,
        latent_dim: int   = CVAE_LATENT_DIM,
        hidden_dim: int   = CVAE_HIDDEN_DIM,
        beta:       float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta

        self.condition_embedder = ConditionEmbedder(embed_dim)
        self.encoder            = CVAEEncoder(latent_dim, hidden_dim)
        self.decoder            = CVAEDecoder(latent_dim, hidden_dim)

    # reparameterisation trick 

    def reparameterise(
        self,
        mu:      torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # forward pass 

    def forward(
        self,
        date_raw:   torch.Tensor,   # (batch, 3)  raw integers as float
        conditions: torch.Tensor,   # (batch, 4)  int indices
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
       
        device = date_raw.device

        # Normalise input date to [0,1]
        offsets = DATE_OFFSETS.to(device)
        ranges  = DATE_RANGES .to(device)
        date_norm = (date_raw.float() - offsets) / ranges   # (batch, 3) in [0,1]

        cond_emb = self.condition_embedder(conditions)       # (batch, 64)
        mu, log_var = self.encoder(date_norm, cond_emb)
        z           = self.reparameterise(mu, log_var)
        recon_norm  = self.decoder(z, cond_emb)              # (batch, 3) in [0,1]

        return recon_norm, mu, log_var, date_norm

    # loss computation 

    def compute_loss(
        self,
        date_raw:   torch.Tensor,
        conditions: torch.Tensor,
        beta:       float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Returns (total_loss, reconstruction_loss, kl_loss).
        # MSE is computed on the [0,1]-normalised scale so day/month/year
        # contribute equally despite having different absolute ranges.
        
        if beta is None:
            beta = self.beta

        recon_norm, mu, log_var, date_norm = self.forward(date_raw, conditions)

        # Reconstruction: MSE on normalised values
        recon_loss = nn.functional.mse_loss(recon_norm, date_norm)

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )

        total = recon_loss + beta * kl_loss
        return total, recon_loss, kl_loss

    # inference 

    @torch.no_grad()
    def sample(
        self,
        conditions: torch.Tensor,   # (batch, 4)
        num_samples: int = 1,
    ) -> torch.Tensor:
        
        # Draw num_samples candidate dates per condition row.
        # Returns (batch, num_samples, 3) as [0,1]-normalised values.
        
        device     = conditions.device
        batch_size = conditions.shape[0]

        cond_emb = self.condition_embedder(conditions)                      # (batch, 64)
        cond_rep = cond_emb.repeat_interleave(num_samples, dim=0)           # (batch*S, 64)

        z    = torch.randn(batch_size * num_samples, self.latent_dim, device=device)
        pred = self.decoder(z, cond_rep)                                    # (batch*S, 3)

        return pred.view(batch_size, num_samples, DATE_DIM)