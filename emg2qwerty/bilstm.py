"""Alternative encoder architectures for emg2qwerty CTC decoding.

All modules share the same SpectrogramNorm + MultiBandRotationInvariantMLP
preprocessing front-end as TDSConvCTCModule. Only the sequence encoder
differs, enabling controlled architecture comparisons.
"""

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
)


class _BaseCTCModule(pl.LightningModule):
    """Shared boilerplate for CTC-based modules with the standard
    preprocessing front-end."""

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def _build_preprocess(self, in_features: int, mlp_features: Sequence[int]):
        return nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

    def _build_metrics(self):
        metrics = MetricCollection([CharacterErrorRates()])
        return nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ============================================================================
# BiLSTM
# ============================================================================

class BiLSTMCTCModule(_BaseCTCModule):
    """Bidirectional LSTM encoder."""

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.preprocess = self._build_preprocess(in_features, mlp_features)

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        self.metrics = self._build_metrics()

    def _run_rnn(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        self.lstm.flatten_parameters()
        if self.training:
            x, _ = self.lstm(x)
        else:
            with torch.backends.cudnn.flags(enabled=False):
                x, _ = self.lstm(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(inputs)
        x = self._run_rnn(x)
        return self.classifier(x)


# ============================================================================
# BiGRU
# ============================================================================

class BiGRUCTCModule(_BaseCTCModule):
    """Bidirectional GRU encoder. Lighter than BiLSTM (no cell state),
    often converges faster on shorter sequences."""

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.preprocess = self._build_preprocess(in_features, mlp_features)

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        self.metrics = self._build_metrics()

    def _run_rnn(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        self.gru.flatten_parameters()
        if self.training:
            x, _ = self.gru(x)
        else:
            with torch.backends.cudnn.flags(enabled=False):
                x, _ = self.gru(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(inputs)
        x = self._run_rnn(x)
        return self.classifier(x)


# ============================================================================
# Transformer
# ============================================================================

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer input.

    Dynamically extends the buffer at inference time when the input
    exceeds the pre-allocated length."""

    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self._build_pe(max_len, d_model))

    @staticmethod
    def _build_pe(length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # (length, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        if T > self.pe.size(0):
            self.pe = self._build_pe(T, self.d_model).to(x.device)
        x = x + self.pe[:T]
        return self.dropout(x)


class TransformerCTCModule(_BaseCTCModule):
    """Transformer encoder with sinusoidal positional encoding."""

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.preprocess = self._build_preprocess(in_features, mlp_features)

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)
        self.metrics = self._build_metrics()

    def _encode_chunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(inputs)
        T = x.size(0)
        # 8000 raw samples / hop_length 16 = 500 spectrogram frames
        CHUNK = 500
        STRIDE = 400
        if not self.training and T > CHUNK:
            outputs = torch.zeros(T, x.size(1), self.classifier[0].in_features,
                                  device=x.device, dtype=x.dtype)
            counts = torch.zeros(T, 1, 1, device=x.device)
            for start in range(0, T, STRIDE):
                end = min(start + CHUNK, T)
                chunk = x[start:end]
                enc = self._encode_chunk(chunk)
                outputs[start:end] += enc
                counts[start:end] += 1
            x = outputs / counts.clamp(min=1)
        else:
            x = self._encode_chunk(x)
        return self.classifier(x)
