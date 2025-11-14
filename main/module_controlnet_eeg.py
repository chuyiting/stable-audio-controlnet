import math
from typing import List, Optional, Literal

import pytorch_lightning as pl
import torch

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from stable_audio_tools.inference.generation import generate_diffusion_cond

from main.controlnet.pretrained import get_pretrained_controlnet_model
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from torch.utils.data import DataLoader
from main.utils import log_wandb_audio_batch, log_wandb_audio_spectrogram,  log_wandb_eeg_batch




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
        cfg_dropout_prob: float
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay

        self.timestep_sampler = "logit_normal"
        self.diffusion_objective = "v"
        model, model_config = get_pretrained_controlnet_model("stabilityai/stable-audio-open-1.0",
                                                              controlnet_types=["eeg"],
                                                              depth_factor=depth_factor)
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


    def configure_optimizers(self):
        params = list(self.model.model.controlnet.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def _unpack_batch(self, batch):
        x_audio = batch["audio"]          # tensor (B, 2, Ta) sr = 44100
        prompts = batch["prompt"]         # List[str] (default collate)
        start_seconds = batch["start_seconds"] # (B,)
        total_seconds = batch["total_seconds"] # (B,)
        eeg = batch['eeg'] # tensor (B, 32, Teeg) sr=128

        device = self.device
        start_seconds = start_seconds.to(device)
        total_seconds = total_seconds.to(device)

        x_audio = x_audio.to(device)
        eeg = eeg.to(device)

        return eeg, x_audio, prompts, start_seconds, total_seconds

    def _sample_timesteps(self, batch_size: int, device: torch.device):
        if self.timestep_sampler == "logit_normal":
            return torch.sigmoid(torch.randn(batch_size, device=device))
        raise ValueError(f"Unknown time step sampler: {self.timestep_sampler}")

    def step(self, batch):
        eeg, x_audio, prompts, start_seconds, total_seconds = self._unpack_batch(batch)
        device = self.device
        print(f"x shape: {x.shape}")
        print(f"total seconds: {total_seconds}")

        # encode to diffusion latent
        diffusion_input = self.model.pretransform.encode(x_audio)  # shape (B, ...)

        # timesteps
        t = self._sample_timesteps(diffusion_input.shape[0], device)

        # alphas/sigmas
        if self.diffusion_objective != "v":
            raise ValueError("Diffusion objective not supported (expected 'v').")
        alphas, sigmas = get_alphas_sigmas(t)  # (B,)

        # broadcast to latent shape (handles 1D/2D/3D etc.)
        while alphas.ndim < diffusion_input.ndim:
            alphas = alphas.unsqueeze(-1)
            sigmas = sigmas.unsqueeze(-1)
        alphas = alphas.to(device)
        sigmas = sigmas.to(device)

        # noise/noised input and v-target
        noise = torch.randn_like(diffusion_input, device=device)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        targets = noise * alphas - diffusion_input * sigmas  # v-prediction target

        # conditioner items per-sample
        B = diffusion_input.shape[0]
        cond_items = []
        for i in range(B):
            item = {
                "prompt": prompts[i] if isinstance(prompts, list) else (prompts[i].item() if torch.is_tensor(prompts) else prompts[i]),
                "seconds_start": start_seconds[i].item() if start_seconds.ndim > 0 else float(start_seconds),
                "seconds_total": total_seconds[i].item() if total_seconds.ndim > 0 else float(total_seconds),
                "eeg": eeg[i:i+1]
            }
            cond_items.append(item)

        cond = self.model.conditioner(cond_items, device=device)

        print(f"x latent shape: {noised_inputs.shape}")
        print(f"eeg shape {cond['eeg'][0].shape}")

        # forward
        output = self.model(
            x=noised_inputs,
            t=t,
            cond=cond,
            cfg_dropout_prob=self.cfg_dropout_prob,
        )

        loss = torch.nn.functional.mse_loss(output, targets).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=self._infer_batch_size(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self._infer_batch_size(batch))
        return loss

    # optional helper so logs have correct batch_size even with dicts
    def _infer_batch_size(self, batch):
        if isinstance(batch, dict) and "audio" in batch and torch.is_tensor(batch["audio"]):
            return batch["audio"].shape[0]
        if isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
            return batch[0].shape[0]
        return None



""" Datamodule """

class WebDatasetDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size_train: int,
        batch_size_val: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn = None,
        drop_last: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn"

    ) -> None:
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
     
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    if hasattr(trainer, "loggers") and trainer.loggers:
        for lg in trainer.loggers:
            if isinstance(lg, WandbLogger):
                return lg

    lg = getattr(trainer, "logger", None)
    if isinstance(lg, WandbLogger):
        return lg
    if isinstance(lg, LoggerCollection):
        for x in lg:
            if isinstance(x, WandbLogger):
                return x
    return None


class SampleLogger(Callback):
    def __init__(
        self,
        sampling_steps: List[int],
        cfg_scale: float,
        num_samples: int = 1
    ) -> None:
        self.sampling_steps = sampling_steps
        self.cfg_scale = cfg_scale
        self.num_samples = num_samples
        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
        wandb_logger = get_wandb_logger(trainer).experiment

        x_audio = batch["audio"]          # tensor (B, 2, Ta) sr = 44100
        prompts = batch["prompt"]         # List[str] (default collate)
        start_seconds = batch["start_seconds"] # (B,)
        total_seconds = batch["total_seconds"] # (B,)
        eeg = batch['eeg'] # tensor (B, 32, Teeg) sr=128

        num_samples = min(self.num_samples, eeg.shape[0])

        conditioning = [{
            "eeg": eeg[i:i+1].to(pl_module.device),
            "prompt": prompts[i],
            "seconds_start": start_seconds[i],
            "seconds_total": total_seconds[i],
        } for i in range(num_samples)]


        for i in range(num_samples):
            log_wandb_eeg_batch(
                logger=wandb_logger,
                id=f"true_{i}",
                samples=eeg[i:i+1],
                sampling_rate=pl_module.sample_rate,
                caption=f"Prompt: {prompts[i]}",
            )

        for steps in self.sampling_steps:

            output = generate_diffusion_cond(
                pl_module.model,
                batch_size=num_samples,
                steps=steps,
                cfg_scale=7.0,
                conditioning=conditioning,
                sample_size=pl_module.sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device="cuda"
            )
            for i in range(num_samples):
                log_wandb_audio_batch(
                    logger=wandb_logger,
                    id=f"sample_x_{i}",
                    samples=output[i:i + 1],
                    sampling_rate=pl_module.sample_rate,
                    caption=f"Sampled in {steps} steps.",
                )
                log_wandb_audio_spectrogram(
                    logger=wandb_logger,
                    id=f"sample_x_{i}",
                    samples=output[i:i + 1],
                    sampling_rate=pl_module.sample_rate,
                    caption=f"Sampled in {steps} steps.",
                )

        if is_train:
            pl_module.train()

