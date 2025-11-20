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

# NOTE: For testing without stable_audio_tools, we inherit from nn.Module
# When stable_audio_tools is installed, uncomment the following lines:
# from stable_audio_tools.models.conditioners import Conditioner
# class EEGConditioner(Conditioner):
#     def __init__(self, ...):
#         super().__init__(output_dim, output_dim)  # Use this instead of super().__init__()

class EEGConditioner(nn.Module):
    """
    EEG Conditioner that uses BIOT encoder to transform EEG signals into conditioning embeddings.
    
    Architecture:
        1. BIOTEncoder: Transforms raw EEG (B, n_channels, T_eeg) -> (B, emb_size)
        2. Temporal Projector: Projects BIOT embeddings to output_dim
        3. Final reshape: (B, output_dim) -> (B, output_dim, 1) for controlnet compatibility
    
    Args:
        output_dim: Output dimension for controlnet conditioning (typically latent_dim from pretransform)
        ckpt_path: Path to pretrained BIOT checkpoint (.ckpt file)
        n_channels: Number of EEG channels (default: 32 for DEAP dataset)
        emb_size: BIOT embedding dimension (default: 256)
        heads: Number of attention heads in BIOT (default: 8)
        depth: Number of transformer layers in BIOT (default: 4)
        n_fft: FFT size for STFT in BIOT (default: 200)
        hop_length: Hop length for STFT in BIOT (default: 100)
        freeze_encoder: Whether to freeze BIOT encoder weights (default: True)
    """
    
    def __init__(
        self,
        output_dim: int, 
        ckpt_path: str,
        n_channels: int = 16,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_fft: int = 200,
        hop_length: int = 100,
    ):
        super().__init__()  # Initialize nn.Module
        
        self.output_dim = output_dim
        self.emb_size = emb_size
        self.n_channels = n_channels
        self.freeze_encoder = freeze_encoder
        
        # Initialize BIOT Encoder
        self.encoder = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Load pretrained weights if checkpoint path is provided
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_pretrained_weights(ckpt_path)
        elif ckpt_path:
            print(f"Warning: Checkpoint path provided but file not found: {ckpt_path}")
        
        # Projection layer: BIOT embedding -> controlnet output dimension
        # BIOT outputs (B, emb_size), we need (B, output_dim, 1)
        self.projector = nn.Sequential(
            nn.Linear(emb_size, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def _load_pretrained_weights(self, ckpt_path: str):
        """Load pretrained BIOT weights from checkpoint."""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Try to find the state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights (strict=False to allow for missing keys in projection layer)
            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded BIOT weights from {ckpt_path}")
        except Exception as e:
            print(f"Error loading checkpoint from {ckpt_path}: {e}")
            print("Continuing with randomly initialized encoder weights")
    
    def forward(self, x, device=None):
        """
        Forward pass through EEG conditioner.
        
        Args:
            x: List of EEG tensors, each of shape (n_channels, T_eeg)
               - List length: B (batch size)
               - Each tensor: (1, n_channels, T_eeg)
               - n_channels: number of EEG channels (e.g., 32 for DEAP)
               - T_eeg: temporal length of EEG signal
            device: Device to move tensors to (optional)
        
        Returns:
            output: Conditioning tensor of shape (B, output_dim, 1) for controlnet
            mask: Mask tensor of shape (B,) with all ones
        """
        # Convert list of tensors to batched tensor
        # Input: List[B] of (n_channels, T_eeg)
        # Output: (B, n_channels, T_eeg)
        if isinstance(x, list):
            # Stack list of tensors into batch
            x = torch.cat(x, dim=0)  # (B, n_channels, T_eeg)
        elif isinstance(x, torch.Tensor):
            # If already a tensor, ensure it's the right shape
            if x.dim() == 2:
                # Single sample: (n_channels, T_eeg) -> (1, n_channels, T_eeg)
                x = x.unsqueeze(0)
            # If already 3D, assume it's (B, n_channels, T_eeg) and use as-is
        
        # Move to device if specified
        if device is not None:
            x = x.to(device)
        
        B = x.shape[0]  # Get batch size
        
        # Ensure encoder is in correct mode
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                # BIOT Encoder: (B, n_channels, T_eeg) -> (B, emb_size)
                eeg_embedding = self.encoder(x)
        else:
            # BIOT Encoder: (B, n_channels, T_eeg) -> (B, emb_size)
            eeg_embedding = self.encoder(x)
        
        # Project to output dimension: (B, emb_size) -> (B, output_dim)
        projected = self.projector(eeg_embedding)
        
        # Reshape for controlnet: (B, output_dim) -> (B, output_dim, 1)
        output = projected.unsqueeze(-1)
        
        # Create mask: (B,) with all ones
        mask = torch.ones(B, device=output.device, dtype=torch.float32).unsqueeze(-1)
        
        return output, mask


        


