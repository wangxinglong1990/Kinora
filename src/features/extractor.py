#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pickle
import re
import zlib
import os

import numpy as np
import torch
import torch.nn as nn

from config import (
    PROTEIN_ESMC_MODEL_NAME,
    SMILES_TRANSFORMER_CHECKPOINT,
    SMILES_TRANSFORMER_DIR,
)

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4
MAX_LEN = 220
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SMILES_TOKENIZER = re.compile(r"(\[[^\[\]]{1,10}\])")


def _norm_text(s):
    return str(s).strip() if s is not None else ""


def _safe_ratio(x, total):
    return float(x) / float(total) if total > 0 else 0.0


def _clean_protein_sequence(seq):
    seq = _norm_text(seq).upper()
    return "".join(ch for ch in seq if ch in AMINO_ACIDS)


def _split_smiles(smiles_text):
    text = _norm_text(smiles_text)
    out = []
    for part in SMILES_TOKENIZER.split(text):
        if not part:
            continue
        if SMILES_TOKENIZER.match(part):
            out.append(part)
        else:
            out.extend(list(part))
    return out


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = _PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.out = nn.Linear(hidden_size, out_size)

    def _encode(self, src):
        embedded = self.embed(src)
        embedded = self.pe(embedded)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)
        penul = output.detach().cpu().numpy()
        output = self.trfm.encoder.layers[-1](output, None)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)
        output = output.detach().cpu().numpy()
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])

    def encode(self, src):
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        st, ed = 0, 100
        out = self._encode(src[:, st:ed])
        while ed < batch_size:
            st += 100
            ed += 100
            out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
        return out


def _extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict) and "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
        state_dict = raw_obj["state_dict"]
    elif isinstance(raw_obj, dict):
        state_dict = raw_obj
    else:
        raise ValueError("Unsupported SMILES_Transform checkpoint format.")
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


class _SmilesTransformerEncoder:
    def __init__(self):
        if not SMILES_TRANSFORMER_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"SMILES_Transform checkpoint not found: {SMILES_TRANSFORMER_CHECKPOINT}. "
                "The current implementation requires this model for substrate encoding."
            )
        raw = torch.load(str(SMILES_TRANSFORMER_CHECKPOINT), map_location="cpu")
        state_dict = _extract_state_dict(raw)

        embed_w = state_dict.get("embed.weight")
        out_w = state_dict.get("out.weight")
        if embed_w is None or out_w is None:
            raise ValueError("SMILES_Transform checkpoint is missing embed.weight or out.weight.")

        self.vocab_size = int(embed_w.shape[0])
        self.hidden_size = int(embed_w.shape[1])
        self.out_size = int(out_w.shape[0])

        layer_ids = set()
        for k in state_dict:
            prefix = "trfm.encoder.layers."
            if k.startswith(prefix):
                idx = k[len(prefix) :].split(".", 1)[0]
                if idx.isdigit():
                    layer_ids.add(int(idx))
        n_layers = max(layer_ids) + 1 if layer_ids else 4

        self.model = _TrfmSeq2seq(self.vocab_size, self.hidden_size, self.out_size, n_layers=n_layers)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.vocab_stoi = self._load_vocab_stoi()

    def _load_vocab_stoi(self):
        vocab_path = SMILES_TRANSFORMER_DIR / "vocab.pkl"
        if not vocab_path.exists():
            return None
        try:
            with open(vocab_path, "rb") as f:
                vocab_obj = pickle.load(f)
            stoi = getattr(vocab_obj, "stoi", None)
            return stoi if isinstance(stoi, dict) else None
        except Exception:
            return None

    def _token_to_id(self, token):
        if self.vocab_stoi is not None:
            return int(self.vocab_stoi.get(token, UNK))
        if self.vocab_size <= 5:
            return UNK
        return 5 + (zlib.crc32(token.encode("utf-8", errors="ignore")) % (self.vocab_size - 5))

    def _encode_one(self, smi):
        tokens = _split_smiles(smi)
        ids = [self._token_to_id(t) for t in tokens]
        ids = [SOS] + ids[: MAX_LEN - 2] + [EOS]
        if len(ids) < MAX_LEN:
            ids += [PAD] * (MAX_LEN - len(ids))
        return ids

    def encode(self, smiles_list):
        seq_ids = [self._encode_one(sm) for sm in smiles_list]
        src = torch.tensor(np.asarray(seq_ids, dtype=np.int64), dtype=torch.long).t().contiguous()
        with torch.no_grad():
            return self.model.encode(src)


_SMILES_ENCODER = None
_PROTEIN_ENCODER = None


def _get_smiles_encoder():
    global _SMILES_ENCODER
    if _SMILES_ENCODER is None:
        _SMILES_ENCODER = _SmilesTransformerEncoder()
    return _SMILES_ENCODER


class _EsmcProteinEncoder:
    def __init__(self):
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
        except Exception as e:
            raise ImportError(
                "Failed to import `esm` dependency. Please ensure the `esm` package is available."
            ) from e

        self.ESMC = ESMC
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.environ["INFRA_PROVIDER"] = "True"
        # Keep the same loading style as Tt_prediction:
        # ESMC.from_pretrained(config.ESM_MODEL_NAME, device=config.DEVICE)
        self.client = self.ESMC.from_pretrained(PROTEIN_ESMC_MODEL_NAME, device=self.device)

        self.cache = {}

    def _encode_one(self, sequence):
        if sequence in self.cache:
            return self.cache[sequence]
        cleaned = _clean_protein_sequence(sequence)
        if len(cleaned) == 0:
            vec = np.zeros(960, dtype=np.float32)
            self.cache[sequence] = vec
            return vec

        protein = self.ESMProtein(sequence=cleaned)
        protein_tensor = self.client.encode(protein)
        logits_output = self.client.logits(
            protein_tensor,
            self.LogitsConfig(sequence=True, return_embeddings=True),
        )
        embedding = logits_output.embeddings
        if embedding.dim() == 3:
            pooled = embedding.mean(dim=1).squeeze(0)
        elif embedding.dim() == 2:
            pooled = embedding.mean(dim=0)
        else:
            pooled = embedding.reshape(-1)
        vec = pooled.detach().cpu().numpy().astype(np.float32)
        self.cache[sequence] = vec
        return vec

    def encode(self, sequences):
        return np.asarray([self._encode_one(seq) for seq in sequences], dtype=np.float32)


def _get_protein_encoder():
    global _PROTEIN_ENCODER
    if _PROTEIN_ENCODER is None:
        _PROTEIN_ENCODER = _EsmcProteinEncoder()
    return _PROTEIN_ENCODER


def extract_joint_features(smiles_list, protein_list):
    if len(smiles_list) != len(protein_list):
        raise ValueError("Length mismatch: smiles_list and protein_list must have the same size.")

    protein_encoder = _get_protein_encoder()
    smiles_encoder = _get_smiles_encoder()
    protein_embeddings = protein_encoder.encode(protein_list).astype(np.float32)
    smiles_embeddings = smiles_encoder.encode(smiles_list).astype(np.float32)

    rows = []
    for i in range(len(protein_list)):
        p_feat = protein_embeddings[i]
        s_feat = smiles_embeddings[i]
        rows.append(np.concatenate([p_feat, s_feat], axis=0))
    return np.asarray(rows, dtype=np.float32)

