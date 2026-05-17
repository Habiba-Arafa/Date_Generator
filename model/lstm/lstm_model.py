from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.config import (
    NUM_DAYS_OF_WEEK, NUM_MONTHS, NUM_LEAP_OPTIONS, NUM_DECADES,
    EMBEDDING_DIM, CONDITION_DIM,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
)

# Vocabulary 
DIGIT_VOCAB_SIZE = 10    # 0-9
SOS_TOKEN        = 10    # start-of-sequence  (fed as first input)
EOS_TOKEN        = 11    # end-of-sequence    (target after last real token)
VOCAB_SIZE       = 12    # total tokens

SEQ_LEN          = 8    # 8 digit-positions per date (dd, mm, yyyy → 2+2+4)


# Helpers: date ↔ digit sequence 

def date_to_digit_sequence(day: int, month: int, year: int) -> List[int]:
    # Convert (day, month, year) to an 8-digit list, zero-padded
   
    day_s   = f"{int(day):02d}"
    month_s = f"{int(month):02d}"
    year_s  = f"{int(year):04d}"
    return [int(c) for c in day_s + month_s + year_s]   # length 8


def digit_sequence_to_date(digits: List[int]) -> Tuple[int, int, int]:
    # Inverse of date_to_digit_sequence. Handles lists of length < 8 gracefully.
    digits = (digits + [0] * 8)[:8]    # pad or truncate to 8
    day   = digits[0] * 10 + digits[1]
    month = digits[2] * 10 + digits[3]
    year  = digits[4] * 1000 + digits[5] * 100 + digits[6] * 10 + digits[7]
    return day, month, year


# Condition embedder 

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
        return torch.cat([d, m, l, e], dim=1)   # (batch, CONDITION_DIM)


# LSTM Date Generator 

class LSTMDateGenerator(nn.Module):
    # Auto-regressive LSTM that generates one digit per step.
    # The condition vector is used to initialise BOTH h0 and c0 of the LSTM
    # (projection layer), so every generated digit is conditioned on the input.

    
    def __init__(
        self,
        hidden_dim:  int = LSTM_HIDDEN_DIM,
        num_layers:  int = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
        embed_dim:   int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Condition path
        self.condition_embedder = ConditionEmbedder(embed_dim)
        # Project condition embedding to (num_layers * hidden_dim)  for h0, c0
        self.h0_projection = nn.Linear(CONDITION_DIM, num_layers * hidden_dim)
        self.c0_projection = nn.Linear(CONDITION_DIM, num_layers * hidden_dim)

        # Token embedding
        self.token_embed = nn.Embedding(VOCAB_SIZE, embed_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Output projection: hidden to digit logits
        self.output_head = nn.Linear(hidden_dim, VOCAB_SIZE)

    # Initial hidden state from conditions 

    def _init_hidden(
        self,
        conditions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map condition embedding to LSTM (h0, c0).
        # Returns h0, c0 each of shape (num_layers, batch, hidden_dim).

        cond_emb = self.condition_embedder(conditions)  # (batch, CONDITION_DIM)
        batch    = cond_emb.size(0)

        h0 = self.h0_projection(cond_emb)              # (batch, num_layers * hidden)
        c0 = self.c0_projection(cond_emb)

        h0 = h0.view(batch, self.num_layers, self.hidden_dim)
        c0 = c0.view(batch, self.num_layers, self.hidden_dim)

        # LSTM expects (num_layers, batch, hidden)
        return h0.permute(1, 0, 2).contiguous(), c0.permute(1, 0, 2).contiguous()

    # Teacher-forcing forward (training) 

    def forward(
        self,
        conditions:   torch.Tensor,   # (batch, 4)  int
        target_digits: torch.Tensor,  # (batch, SEQ_LEN)  int  – ground truth
    ) -> torch.Tensor:

        # Teacher-forcing: feed the *ground-truth* previous token at each step.
        # Returns logits of shape (batch, SEQ_LEN, VOCAB_SIZE).

        batch = conditions.size(0)
        h0, c0 = self._init_hidden(conditions)

        # Input sequence: [SOS, d0, d1, d2, …, d6]  (drop last digit; predict it)
        sos = torch.full((batch, 1), SOS_TOKEN, dtype=torch.long, device=conditions.device)
        input_seq = torch.cat([sos, target_digits[:, :-1]], dim=1)   # (batch, SEQ_LEN)

        emb    = self.token_embed(input_seq)    # (batch, SEQ_LEN, embed_dim)
        out, _ = self.lstm(emb, (h0, c0))      # (batch, SEQ_LEN, hidden_dim)
        logits = self.output_head(out)          # (batch, SEQ_LEN, VOCAB_SIZE)
        return logits

    # Auto-regressive inference 

    @torch.no_grad()
    def generate(
        self,
        conditions:   torch.Tensor,    # (batch, 4)
        temperature:  float = 1.0,
        greedy:       bool  = False,
    ) -> torch.Tensor:
        
        # Generate a digit sequence auto-regressively.
        # Returns (batch, SEQ_LEN) int tensor.
        
        batch  = conditions.size(0)
        device = conditions.device
        h, c   = self._init_hidden(conditions)

        generated = []
        current_token = torch.full((batch, 1), SOS_TOKEN, dtype=torch.long, device=device)

        for _ in range(SEQ_LEN):
            emb       = self.token_embed(current_token)         # (batch, 1, embed_dim)
            out, (h, c) = self.lstm(emb, (h, c))               # (batch, 1, hidden)
            logits    = self.output_head(out.squeeze(1))        # (batch, VOCAB_SIZE)

            # Only allow digit tokens 0-9 (mask SOS and EOS)
            logits[:, SOS_TOKEN] = -1e9
            logits[:, EOS_TOKEN] = -1e9

            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs      = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)
            current_token = next_token

        return torch.cat(generated, dim=1)   # (batch, SEQ_LEN)

    #  Convenience: generate multiple samples per condition 

    @torch.no_grad()
    def generate_samples(
        self,
        conditions:  torch.Tensor,
        num_samples: int   = 1,
        temperature: float = 1.2,
    ) -> torch.Tensor:
        # Returns (batch, num_samples, SEQ_LEN) int tensor.
        
        batch  = conditions.size(0)
        device = conditions.device
        cond_rep = conditions.repeat_interleave(num_samples, dim=0)  # (batch*S, 4)
        digits   = self.generate(cond_rep, temperature=temperature)  # (batch*S, 8)
        return digits.view(batch, num_samples, SEQ_LEN)