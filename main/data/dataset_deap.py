import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample
from torch.utils.data import Dataset


@dataclass
class _DEAPSampleDescriptor:
    """Lightweight handle to a single DEAP trial."""

    subject_id: int
    trial_index: int
    eeg_file: str
    audio_path: str


class DEAPAudioEEGDataset(Dataset):
    """
    Loads aligned audio snippets and EEG control signals derived from the DEAP dataset.

    Args:
        eeg_root: Directory that contains the preprocessed DEAP `.dat` files (e.g. `data_preprocessed_python`).
        audio_root: Directory that contains one audio file per trial.
        audio_template: Filename pattern used to look up audio files. The template receives
            `subject_id` (1-indexed) and `trial_index` (1-indexed) and must point to a file under `audio_root`.
        sample_rate: Desired audio sample rate fed to the diffusion model.
        chunk_dur: Length (in seconds) for both the audio and EEG crops that are returned.
        eeg_sample_rate: Sampling rate of the EEG signals. Defaults to 128 Hz for DEAP.
        num_eeg_channels: Number of EEG channels to keep (first 32 for DEAP).
        baseline_seconds: Amount of baseline data to discard from the beginning of every trial.
        cache_subjects: When True the full EEG tensor for a subject is kept in memory after the first access.
            This speeds up iteration at the cost of roughly ~40 MB per subject.
    """

    def __init__(
        self,
        eeg_root: str,
        audio_root: str,
        audio_template: str,
        sample_rate: int,
        chunk_dur: float,
        eeg_sample_rate: int = 128,
        num_eeg_channels: int = 32,
        baseline_seconds: float = 3.0,
        cache_subjects: bool = False,
        stereo: bool = True,
    ) -> None:
        super().__init__()
        self.eeg_root = Path(eeg_root)
        self.audio_root = Path(audio_root)
        self.audio_template = audio_template
        self.sample_rate = sample_rate
        self.chunk_dur = chunk_dur
        self.eeg_sample_rate = eeg_sample_rate
        self.num_eeg_channels = num_eeg_channels
        self.baseline_seconds = baseline_seconds
        self.cache_subjects = cache_subjects
        self.stereo = stereo

        if not self.eeg_root.exists():
            raise FileNotFoundError(f"EEG root '{self.eeg_root}' was not found.")
        if not self.audio_root.exists():
            raise FileNotFoundError(f"Audio root '{self.audio_root}' was not found.")

        self.chunk_samples = int(sample_rate * chunk_dur)
        self.chunk_eeg_samples = max(1, int(self.eeg_sample_rate * chunk_dur))
        self.baseline_samples = int(self.baseline_seconds * self.eeg_sample_rate)

        self._subjects = sorted(self.eeg_root.glob("s*.dat"))
        if not self._subjects:
            raise RuntimeError(f"No DEAP subject files found under '{self.eeg_root}'.")

        self._sample_descriptors: List[_DEAPSampleDescriptor] = []
        for eeg_path in self._subjects:
            subject_id = int(eeg_path.stem[1:])
            with open(eeg_path, "rb") as f:
                subject_data = pickle.load(f, encoding="latin1")
            eeg_trials = subject_data["data"]
            labels = subject_data["labels"]
            trial_count = eeg_trials.shape[0]
            for trial_idx in range(trial_count):
                audio_path = self._format_audio_path(subject_id, trial_idx + 1)
                if not audio_path.exists():
                    continue
                self._sample_descriptors.append(
                    _DEAPSampleDescriptor(
                        subject_id=subject_id,
                        trial_index=trial_idx,
                        eeg_file=str(eeg_path),
                        audio_path=str(audio_path),
                    )
                )

        if not self._sample_descriptors:
            raise RuntimeError(
                f"No valid DEAP trials found. Ensure audio files exist in '{self.audio_root}'."
            )

        self._subject_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self._sample_descriptors)

    def __getitem__(self, index: int):
        desc = self._sample_descriptors[index]
        eeg_trials, label_tensor = self._load_subject_trials(desc.eeg_file)
        eeg = eeg_trials[desc.trial_index]
        labels = label_tensor[desc.trial_index]

        audio, audio_sr = torchaudio.load(desc.audio_path)
        if audio_sr != self.sample_rate:
            audio = resample(audio, orig_freq=audio_sr, new_freq=self.sample_rate)
        if self.stereo and audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        if audio.shape[0] > 2:
            audio = audio[:2]

        audio_chunk, start_seconds = self._sample_audio_chunk(audio, self.sample_rate)
        eeg_chunk = self._sample_eeg_chunk(eeg, start_seconds)

        prompt = (
            f"subject: s{desc.subject_id:02d}; trial: {desc.trial_index + 1:02d}; "
            f"valence: {labels[0]:.2f}; arousal: {labels[1]:.2f}; "
            f"dominance: {labels[2]:.2f}; liking: {labels[3]:.2f}"
        )

        total_seconds = audio_chunk.shape[-1] / self.sample_rate
        return audio_chunk, eeg_chunk, prompt, start_seconds, total_seconds

    def _format_audio_path(self, subject_id: int, trial_index_one_based: int) -> Path:
        filename = self.audio_template.format(subject_id=subject_id, trial_index=trial_index_one_based)
        return self.audio_root / filename

    def _load_subject_trials(self, eeg_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_subjects and eeg_file in self._subject_cache:
            return self._subject_cache[eeg_file]

        with open(eeg_file, "rb") as f:
            subject_data = pickle.load(f, encoding="latin1")
        eeg_trials = torch.tensor(
            subject_data["data"][:, : self.num_eeg_channels, self.baseline_samples :],
            dtype=torch.float32,
        )
        labels = torch.tensor(subject_data["labels"], dtype=torch.float32)

        if self.cache_subjects:
            self._subject_cache[eeg_file] = (eeg_trials, labels)
        return eeg_trials, labels

    def _sample_audio_chunk(self, audio: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, float]:
        total_samples = audio.shape[-1]
        if total_samples >= self.chunk_samples:
            max_start = total_samples - self.chunk_samples
            start = random.randint(0, max_start) if max_start > 0 else 0
            chunk = audio[:, start : start + self.chunk_samples]
            start_seconds = start / sample_rate
        else:
            pad_amount = self.chunk_samples - total_samples
            chunk = F.pad(audio, (0, pad_amount))
            start_seconds = 0.0
        return chunk, start_seconds

    def _sample_eeg_chunk(self, eeg: torch.Tensor, start_seconds: float) -> torch.Tensor:
        start = int(start_seconds * self.eeg_sample_rate)
        end = start + self.chunk_eeg_samples
        if eeg.shape[-1] >= end:
            chunk = eeg[:, start:end]
        else:
            chunk = F.pad(eeg[:, start:], (0, max(0, self.chunk_eeg_samples - eeg[:, start:].shape[-1])))
        return chunk


def create_deap_dataset(
    eeg_root: str,
    audio_root: str,
    sample_rate: int,
    chunk_dur: float,
    audio_template: str = "s{subject_id:02d}_t{trial_index:02d}.wav",
    eeg_sample_rate: int = 128,
    num_eeg_channels: int = 32,
    baseline_seconds: float = 3.0,
    cache_subjects: bool = False,
    stereo: bool = True,
) -> DEAPAudioEEGDataset:
    """
    Hydra-friendly factory that instantiates :class:`DEAPAudioEEGDataset`.
    """
    return DEAPAudioEEGDataset(
        eeg_root=eeg_root,
        audio_root=audio_root,
        audio_template=audio_template,
        sample_rate=sample_rate,
        chunk_dur=chunk_dur,
        eeg_sample_rate=eeg_sample_rate,
        num_eeg_channels=num_eeg_channels,
        baseline_seconds=baseline_seconds,
        cache_subjects=cache_subjects,
        stereo=stereo,
    )


def collate_fn(samples: Sequence[Tuple[torch.Tensor, torch.Tensor, str, float, float]]):
    """
    Collates variable-length EEG tensors by zero padding and stacks audio/control batches.
    """
    audio, eeg, prompts, start_seconds, total_seconds = zip(*samples)
    audio_batch = torch.stack(audio)

    eeg_max_len = max(t.shape[-1] for t in eeg)
    eeg_batch = torch.stack([F.pad(t, (0, eeg_max_len - t.shape[-1])) for t in eeg])

    return (
        audio_batch,
        eeg_batch,
        list(prompts),
        list(start_seconds),
        list(total_seconds),
    )
