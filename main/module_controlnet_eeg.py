from typing import Callable, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from stable_audio_tools.inference.generation import generate_diffusion_cond

from main.controlnet.pretrained import get_pretrained_controlnet_model
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from torch.utils.data import DataLoader
from main.utils import log_wandb_audio_batch, log_wandb_audio_spectrogram


class EEGProjector(nn.Module):
    """Maps EEG activations to an audio-like control signal."""

    def __init__(self, num_eeg_channels: int, hidden_dim: int, target_length: int) -> None:
        super().__init__()
        self.target_length = target_length
        self.net = nn.Sequential(
            nn.Conv1d(num_eeg_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 2, kernel_size=1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # Normalize per-channel to keep ranges comparable across subjects.
        eeg = eeg - eeg.mean(dim=-1, keepdim=True)
        eeg = eeg / (eeg.std(dim=-1, keepdim=True) + 1e-6)
        eeg_resampled = F.interpolate(
            eeg,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        )
        return torch.tanh(self.net(eeg_resampled))


""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        depth_factor: float,
        cfg_dropout_prob: float,
        num_eeg_channels: int,
        projector_hidden_dim: int,
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay

        self.timestep_sampler = "logit_normal"
        self.diffusion_objective = "v"
        model, model_config = get_pretrained_controlnet_model(
            "stabilityai/stable-audio-open-1.0",
            controlnet_types=["audio"],
            depth_factor=depth_factor,
        )
        self.model_config = model_config
        self.sample_size = model_config["sample_size"]
        self.sample_rate = model_config["sample_rate"]

        self.cfg_dropout_prob = cfg_dropout_prob

        self.model = model
        self.model.model.model.requires_grad_(False)
        self.model.conditioner.requires_grad_(False)
        self.model.conditioner.eval()
        self.model.pretransform.requires_grad_(False)
        self.model.pretransform.eval()

        self.eeg_projector = EEGProjector(
            num_eeg_channels=num_eeg_channels,
            hidden_dim=projector_hidden_dim,
            target_length=self.sample_size,
        )

    def configure_optimizers(self):
        params = list(self.model.model.controlnet.parameters()) + list(self.eeg_projector.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        if self.timestep_sampler == "logit_normal":
            return torch.sigmoid(torch.randn(batch_size, device=self.device))
        raise ValueError(f"Unknown time step sampler: {self.timestep_sampler}")

    def _build_conditioning(
        self,
        projected_eeg: torch.Tensor,
        prompts: List[str],
        start_seconds: List[float],
        total_seconds: List[float],
    ):
        return [
            {
                "prompt": prompts[i],
                "seconds_start": start_seconds[i],
                "seconds_total": total_seconds[i],
                "audio": projected_eeg[i : i + 1],
            }
            for i in range(projected_eeg.shape[0])
        ]

    def step(self, batch):
        x, eeg, prompts, start_seconds, total_seconds = batch

        diffusion_input = self.model.pretransform.encode(x)
        projected_eeg = self.eeg_projector(eeg.to(self.device))

        t = self._sample_timesteps(x.shape[0])

        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        else:
            raise ValueError("Diffusion objective not supported")

        alphas = alphas[:, None, None].to(self.device)
        sigmas = sigmas[:, None, None].to(self.device)

        noise = torch.randn_like(diffusion_input).to(self.device)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas

        conditioning = self._build_conditioning(projected_eeg, prompts, start_seconds, total_seconds)

        output = self.model(
            x=noised_inputs,
            t=t.to(self.device),
            cond=self.model.conditioner(conditioning, device=self.device),
            cfg_dropout_prob=self.cfg_dropout_prob,
        )
        loss = torch.nn.functional.mse_loss(output, targets).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("valid_loss", loss)
        return loss


""" Datamodule """


class EEGDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size_train: int,
        batch_size_val: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn",
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.shuffle_train = shuffle_train
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=self.shuffle_train,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    print("WandbLogger not found.")
    return None


class SampleLogger(Callback):
    def __init__(
        self,
        sampling_steps: List[int],
        cfg_scale: float,
        num_samples: int = 1,
    ) -> None:
        self.sampling_steps = sampling_steps
        self.cfg_scale = cfg_scale
        self.num_samples = num_samples
        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer)
        if wandb_logger is None:
            if is_train:
                pl_module.train()
            return
        wandb_experiment = wandb_logger.experiment

        x, eeg, prompts, start_seconds, total_seconds = batch
        x = torch.clip(x, -1, 1)
        num_samples = min(self.num_samples, x.shape[0])
        projected = pl_module.eeg_projector(eeg[:num_samples].to(pl_module.device))
        conditioning = pl_module._build_conditioning(
            projected,
            prompts[:num_samples],
            start_seconds[:num_samples],
            total_seconds[:num_samples],
        )

        for i in range(num_samples):
            log_wandb_audio_batch(
                logger=wandb_experiment,
                id=f"target_{i}",
                samples=x[i : i + 1],
                sampling_rate=pl_module.sample_rate,
                caption=f"Prompt: {prompts[i]}",
            )
            log_wandb_audio_spectrogram(
                logger=wandb_experiment,
                id=f"target_{i}",
                samples=x[i : i + 1],
                sampling_rate=pl_module.sample_rate,
                caption=f"Prompt: {prompts[i]}",
            )

        for steps in self.sampling_steps:
            output = generate_diffusion_cond(
                pl_module.model,
                batch_size=num_samples,
                steps=steps,
                cfg_scale=self.cfg_scale,
                conditioning=conditioning,
                sample_size=pl_module.sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=pl_module.device,
            )
            for i in range(num_samples):
                log_wandb_audio_batch(
                    logger=wandb_experiment,
                    id=f"sample_{steps}_{i}",
                    samples=output[i : i + 1],
                    sampling_rate=pl_module.sample_rate,
                    caption=f"Sampled in {steps} steps.",
                )
                log_wandb_audio_spectrogram(
                    logger=wandb_experiment,
                    id=f"sample_{steps}_{i}",
                    samples=output[i : i + 1],
                    sampling_rate=pl_module.sample_rate,
                    caption=f"Sampled in {steps} steps.",
                )

        if is_train:
            pl_module.train()
