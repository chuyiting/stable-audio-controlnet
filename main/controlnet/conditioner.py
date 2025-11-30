import torch
import torch.nn as nn
import os
import sys

# Add BIOT to path
# TODO: Consider making this more robust with proper package installation
BIOT_PATH = os.path.join(os.path.dirname(__file__), '../../../eeg-tutorial/encoders/BIOT')
if BIOT_PATH not in sys.path:
    sys.path.append(BIOT_PATH)

from model.biot import BIOTEncoder
from main.controlnet.eegnet import EEGNet

class EEGConditioner(nn.Module):

    def __init__(
        self,
        output_dim: int,
        ckpt_path: str = None,
        n_channels: int = 32,
        duration_s: float = 10.0,
        project_to_T: bool = True,  
        use_film: bool = False,
        encoder_type: str = 'eegnet',  # 'biot' or 'eegnet'
        post_norm: bool = True,
        # EEGNet parameters
        eeg_freq: int = 128,
        kernel_1: int = 64,
        kernel_2: int = 16,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        # BIOT parameters
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_fft: int = 200,
        hop_length: int = 100,
        # Stable Audio Open specific
        latent_rate_hz: float = 21.5,
    ):
        super().__init__()
        if use_film:
            assert project_to_T == False, "When using FiLM, project_to_T must be False."

        self.output_dim = output_dim
        self.n_channels = n_channels
        self.use_film = use_film
        self.project_to_T = project_to_T
        self.encoder_type = encoder_type
        self.post_norm = post_norm

        self.latent_rate_hz = latent_rate_hz
        self.duration_s = float(duration_s)

        assert duration_s > 0 or use_film, "Either duration_s must be positive."
        self.T_latent = int(self.duration_s * self.latent_rate_hz)
        self.T_eeg = int(self.duration_s * eeg_freq)

        # Initialize BIOT Encoder
        if self.encoder_type == 'eegnet':
            self.encoder = EEGNet(
                chunk_size=self.T_eeg,
                num_electrodes=n_channels,
                F1=F1,
                F2=F2,
                D=D,
                kernel_1=kernel_1,
                kernel_2=kernel_2,
                dropout=0.25,
                duration_s=duration_s,
                latent_rate_hz=self.latent_rate_hz,
                keep_time_dim=self.project_to_T,
                time_last=False
            )
            self.emb_size = F2 if self.project_to_T else self.encoder.feature_dim() 
            assert post_norm == False, "For EEGNet encoder, post_norm must be False."
        else:
            self.encoder = BIOTEncoder(
                emb_size=emb_size,
                heads=heads,
                depth=depth,
                n_channels=n_channels,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            self.emb_size = emb_size

        # Load pretrained weights if checkpoint path is provided
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_pretrained_weights(ckpt_path)
        elif ckpt_path:
            print(f"Warning: Checkpoint path provided but file not found: {ckpt_path}")
        
        if not ckpt_path:
            print("No checkpoint path provided; using randomly initialized encoder weights.")

        if post_norm:
            self.post_encoder_norm = nn.LayerNorm(self.emb_size)
        else:
            self.post_encoder_norm = nn.Identity()

        # eegnet keeps the time dimension, so no need to project to T_latent
        if self.project_to_T and not self.encoder_type == 'eegnet':
            self.projector = nn.Sequential(
                nn.Linear(self.emb_size, self.output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim * self.T_latent),
            )
        else: 
            self.projector = nn.Sequential(
                nn.Linear(self.emb_size, output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim),
            )


    def _load_pretrained_weights(self, ckpt_path: str):
        """Load pretrained BIOT weights from checkpoint."""
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load weights into encoder; projector stays randomly init
            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded BIOT weights from {ckpt_path}")
        except Exception as e:
            print(f"Error loading checkpoint from {ckpt_path}: {e}")
            print("Continuing with randomly initialized encoder weights")

    def forward(self, x, device=None):
        """
        Forward pass through EEG conditioner.

        Args:
            x: EEG input, either
               - list of tensors, each of shape (1, n_channels, T_eeg) or
               - tensor of shape (B, n_channels, T_eeg) or (n_channels, T_eeg)
            device: Optional device to move tensors to.

        Returns:
            output: Conditioning tensor of shape (B, output_dim, T_latent)
            mask:   Mask tensor of shape (B, T_latent)
        """
        if isinstance(x, list):
            x = torch.cat(
                [
                    t if t.dim() == 3 else t.unsqueeze(0)
                    for t in x
                ],
                dim=0,
            )  # (B, n_channels, T_eeg)
        elif isinstance(x, torch.Tensor):
            if x.dim() == 2:
                x = x.unsqueeze(0)

        if device is not None:
            x = x.to(device)

        B = x.shape[0]

        eeg_embedding = self.encoder(x) # (B, emb_size) or (B, T_latent, embed_size)
        eeg_embedding = self.post_encoder_norm(eeg_embedding)

        # Project to (B, output_dim * T_latent) or (B, output_dim) or (B, T_latent, output_dim)
        projected = self.projector(eeg_embedding)  

        if self.use_film:
            output = projected.view(B, self.output_dim)
        elif self.project_to_T and projected.dim() == 2: 
            output = projected.view(B, self.output_dim, self.T_latent)
        elif self.project_to_T and projected.dim() == 3:
            output = projected.permute(0, 2, 1)  # (B, output_dim, T_latent)
        else:
            output = projected.view(B, self.output_dim, 1)

        if self.project_to_T:
            mask = torch.ones(
                B,
                self.T_latent,
                device=output.device,
                dtype=torch.float32,
            )
        else:
            mask = torch.ones(
                B,
                device=output.device,
                dtype=torch.float32,
            )
        
        # print(f"EEGConditioner output shape: {output.shape}, mask shape: {mask.shape}")

        return output, mask

