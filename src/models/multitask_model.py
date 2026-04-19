#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class MultiTaskRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.35,
        use_mha: bool = True,
        attn_tokens: int = 4,
        attn_dim: int = 64,
        attn_heads: int = 4,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_mha = use_mha
        self.use_gate = use_gate
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_dim = hidden_dim // 2

        if self.use_mha:
            self.attn_tokens = attn_tokens
            self.attn_dim = attn_dim
            self.token_proj = nn.Linear(hidden_dim, attn_tokens * attn_dim)
            self.mha = nn.MultiheadAttention(attn_dim, attn_heads, dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(attn_dim)
            self.res_proj = nn.Linear(hidden_dim, attn_dim)
            self.gate_proj = nn.Linear(hidden_dim, 1)
            self.post_attn = nn.Sequential(
                nn.Linear(attn_dim, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.post_attn = nn.Sequential(
                nn.Linear(hidden_dim, head_dim),
                nn.BatchNorm1d(head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.km_head = nn.Linear(head_dim, 1)
        self.kcat_head = nn.Linear(head_dim, 1)

    def forward(self, x):
        shared_feat = self.shared(x)
        if self.use_mha:
            tokens = self.token_proj(shared_feat).view(-1, self.attn_tokens, self.attn_dim)
            attn_out, _ = self.mha(tokens, tokens, tokens, need_weights=False)
            tokens = self.attn_norm(tokens + attn_out)
            pooled = tokens.mean(dim=1)  # [B, attn_dim]
            res = self.res_proj(shared_feat)  # [B, attn_dim]
            if self.use_gate:
                gate = torch.sigmoid(self.gate_proj(shared_feat))  # [B,1]
                pooled = gate * pooled + (1.0 - gate) * res
            else:
                pooled = 0.5 * (pooled + res)
            shared_feat = self.post_attn(pooled)
        else:
            shared_feat = self.post_attn(shared_feat)

        km_pred = self.km_head(shared_feat)
        kcat_pred = self.kcat_head(shared_feat)
        return torch.cat([km_pred, kcat_pred], dim=1)

