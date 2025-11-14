from stable_audio_tools.models.conditioners import Conditioner

# TODO clean this
import sys
sys.path.append('../../../eeg-tutorial/encoders/BIOT')

import torch.nn as nn

import torch
from torch import nn
from stable_audio_tools.models.conditioners import Conditioner


class EEGConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        ckpt_path: str,      # TODO: load pretrained EEG encoder here later
        eeg_t: int = 6086,
        eeg_dim: int = 32,
    ):
        # We choose dim=output_dim because this conditioner already outputs output_dim features
        super().__init__(dim=output_dim, output_dim=output_dim)

        self.eeg_t = eeg_t
        self.eeg_dim = eeg_dim

        # x will be (B, eeg_dim, eeg_t) → flatten to (B, eeg_dim * eeg_t)
        self.projector = nn.Linear(eeg_dim * eeg_t, output_dim)

    def forward(self, x: torch.Tensor, device=None):
        """
        x: list tensors (32, T_eeg) where the length is B

        Returns:
            encoded: (B, output_dim, 1)
            mask:    (B,) all ones (not really used, but matches (tensor, mask) API)
        """
        if device is not None:
            x = x.to(device)

        B = len(x)
        C, T = x[0].shape
        assert C == self.eeg_dim, f"Expected {self.eeg_dim} EEG channels, got {C}"
        assert T == self.eeg_t, f"Expected T_eeg={self.eeg_t}, got {T}"

        x = torch.stack(x, dim=0)
        x_flat = x.reshape(B, C * T)
        encoded = self.projector(x_flat)
        encoded = encoded.unsqueeze(-1)
        mask = torch.ones(B, device=encoded.device, dtype=torch.bool)

        return encoded, mask
