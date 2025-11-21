class EEGConditioner(nn.Module):
    """
    EEG Conditioner that uses BIOT encoder to transform EEG signals into
    time-resolved conditioning embeddings for Stable Audio / ControlNet.

    Architecture:
        1. BIOTEncoder: (B, n_channels, T_eeg) -> (B, emb_size)
        2. Projector: (B, emb_size) -> (B, output_dim * T_latent)
        3. Reshape: (B, output_dim * T_latent) -> (B, output_dim, T_latent)

    Args:
        output_dim: Output channel dimension for controlnet conditioning
                    (typically latent_dim from pretransform).
        ckpt_path: Path to pretrained BIOT checkpoint (.ckpt file).
        n_channels: Number of EEG channels (default: 16).
        emb_size: BIOT embedding dimension (default: 256).
        heads: Number of attention heads in BIOT (default: 8).
        depth: Number of transformer layers in BIOT (default: 4).
        n_fft: FFT size for STFT in BIOT (default: 200).
        hop_length: Hop length for STFT in BIOT (default: 100).
        duration_s: Audio duration in seconds. Used with a fixed latent rate
                    (21.5 Hz) to compute the latent time length.
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
        duration_s: float = -1,   
    ):
        super().__init__()

        self.output_dim = output_dim
        self.emb_size = emb_size
        self.n_channels = n_channels

        # Stable Audio Open latent rate (approx)
        self.latent_rate_hz = 21.5
        self.duration_s = float(duration_s)

        if self.duration_s > 0:
            self.n_latent = int(round(self.duration_s * self.latent_rate_hz))
        else:
            self.n_latent = -1 

        # Initialize BIOT Encoder
        self.encoder = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Load pretrained weights if checkpoint path is provided
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_pretrained_weights(ckpt_path)
        elif ckpt_path:
            print(f"Warning: Checkpoint path provided but file not found: {ckpt_path}")

        self.post_encoder_norm = nn.LayerNorm(emb_size)

        # Projection layer:
        #   (B, emb_size) -> (B, output_dim * n_latent)
        if self.n_latent > 0:
            self.projector = nn.Sequential(
                nn.Linear(emb_size, output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim * self.n_latent),
            )
        else: 
            self.projector = nn.Sequential(
                nn.Linear(emb_size, output_dim * 2),
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
        # Convert list of tensors to batched tensor
        if isinstance(x, list):
            # Expect list length B, each (1, n_channels, T_eeg) or (n_channels, T_eeg)
            x = torch.cat(
                [
                    t if t.dim() == 3 else t.unsqueeze(0)
                    for t in x
                ],
                dim=0,
            )  # (B, n_channels, T_eeg)
        elif isinstance(x, torch.Tensor):
            if x.dim() == 2:
                # Single sample: (n_channels, T_eeg) -> (1, n_channels, T_eeg)
                x = x.unsqueeze(0)
            # if dim == 3, assume (B, n_channels, T_eeg)

        if device is not None:
            x = x.to(device)

        B = x.shape[0]

        # BIOT Encoder: (B, n_channels, T_eeg) -> (B, emb_size)
        eeg_embedding = self.encoder(x)
        eeg_embedding = self.post_encoder_norm(eeg_embedding)

        # Project to (B, output_dim * T_latent)
        projected = self.projector(eeg_embedding)  # (B, output_dim * n_latent)

        # Reshape to (B, output_dim, T_latent)
        if self.n_latent > 0: 
            output = projected.view(B, self.output_dim, self.n_latent)
        else:
             output = projected.view(B, self.output_dim, 1)

        # Mask over time dimension: (B, T_latent)
        if self.n_latent > 0:
            mask = torch.ones(
                B,
                self.n_latent,
                device=output.device,
                dtype=torch.float32,
            )
        else:
            mask = torch.ones(
                B,
                device=output.device,
                dtype=torch.float32,
            )

        return output, mask
