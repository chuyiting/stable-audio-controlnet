from __future__ import annotations
import os
import glob
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    raise RuntimeError("This dataset requires PyTorch. Please `pip install torch`.\n" + str(e))

import torchaudio
import librosa
import soundfile as sf
import math

"""
preprocessed DEAP channels 
source: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
"""
DEAP_CHANNELS = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]

"""
BIOT channels
https://github.com/ycq091044/BIOT/blob/main/datasets/TUAB/process.py
Assumption: A1, A2 = 0
"""
BIOT_PAIRS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "C3", "C4"
]

DEAP_IDX = {ch.upper(): i for i, ch in enumerate(DEAP_CHANNELS)}


def _deap_to_biot_bipolar(eeg, dim: int = 16):
    """
    Convert DEAP 32-channel EEG to BIOT bipolar montage.

    Parameters
    ----------
    eeg : torch.Tensor
        Shape (32, T) or (B, 32, T).
    dim : int
        Number of output channels. Supported values:
        - 16: only the first 16 bipolar channels
        - 18: 16 bipolar channels + C3, C4 as the last two channels

    Returns
    -------
    eeg_bipolar : torch.Tensor
        Shape (dim, T) or (B, dim, T).
        When dim == 18, channels 16 and 17 (0-based) are C3 and C4.
    """
    data = eeg
    assert data.shape[-2] == 32, f"Expected channels dim (-2) = 32, got {data.shape}"
    assert dim in (16, 18), f"dim must be 16 or 18, got {dim}"
    assert dim <= len(BIOT_PAIRS), f"dim {dim} exceeds available BIOT mappings ({len(BIOT_PAIRS)})"

    pairs_to_use = BIOT_PAIRS[:dim]

    out_shape = list(data.shape)
    out_shape[-2] = dim
    eeg_bipolar = data.new_zeros(*out_shape)

    for k, pair in enumerate(pairs_to_use):
        # Bipolar pair: "A-B"
        if "-" in pair:
            a_name, b_name = pair.split("-")
            a_idx = DEAP_IDX[a_name.upper()]
            b_idx = DEAP_IDX[b_name.upper()]
            a_sig = data.select(dim=-2, index=a_idx)
            b_sig = data.select(dim=-2, index=b_idx)
            eeg_bipolar[..., k, :] = a_sig - b_sig
        else:
            # Monopolar channel: e.g., "C3" or "C4"
            ch_idx = DEAP_IDX[pair.upper()]
            ch_sig = data.select(dim=-2, index=ch_idx)
            eeg_bipolar[..., k, :] = ch_sig

    return eeg_bipolar


def _stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return audio


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio

    tensor = torch.from_numpy(audio)
    out = torchaudio.functional.resample(tensor, src_sr, dst_sr)
    return out.numpy()


def _get_audio_len_sec(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def _read_audio_segment(path: str, start_sec: float, dur_sec: float, target_sr: int,
                        force_stereo: bool = True) -> np.ndarray:
    """
    Load [start_sec, start_sec + dur_sec) from a WAV file using torchaudio,
    resample to target_sr, return (C, T) float32. Raises on out-of-bounds or short audio.
    """
    PEAK_LEVEL = 0.98
    wav_t, sr = torchaudio.load(path)  # (C, N) float32
    if force_stereo and wav_t.shape[0] == 1:
        wav_t = wav_t.expand(2, -1)  # duplicate mono if necessary

    s = int(round(max(0.0, float(start_sec)) * sr))
    n_src = int(round(float(dur_sec) * sr))
    n_tgt = int(round(float(dur_sec) * target_sr))

    if s >= wav_t.shape[1]:
        raise ValueError(
            f"start_sec={start_sec} (samples {s}) is beyond file length "
            f"{wav_t.shape[1]/sr:.3f}s for {path}"
        )

    e = s + n_src
    window = wav_t[:, s:e]

    if window.shape[1] < n_src:
        have = window.shape[1] / sr
        need = n_src / sr
        raise ValueError(
            f"Requested duration {dur_sec}s (need {need:.3f}s) exceeds available "
            f"{have:.3f}s from start={start_sec}s in {path}"
        )

    # Resample if needed
    if sr != target_sr:
        window = torchaudio.functional.resample(window, sr, target_sr)

    # Guard tiny resampler drift by hard trim (never pad)
    if window.shape[1] < n_tgt:
        # This should not happen if upstream checks are correct
        raise RuntimeError(
            f"Post-resample shorter than expected: got {window.shape[1]} < {n_tgt} samples for {path}"
        )
    elif window.shape[1] > n_tgt:
        window = window[:, :n_tgt]

    peak_val = window.abs().max()
    if peak_val > 0:
        window = window * (PEAK_LEVEL / peak_val)

    return window.numpy().astype(np.float32, copy=False)


def _is_classical_audio_id(audio_id: str) -> bool:
    """
    Classical music files are single-digit numbers according to the notes.
    """
    audio_id = audio_id.strip()
    return audio_id.isdigit() and len(audio_id) == 1


@dataclass
class _DallyIdx:
    audio_id: str
    subject: str       # "01".."21"
    task: str          # "musicFt", "justMusic", "justFt", etc.
    start_sec: float
    dur_sec: float
    eeg_slice: Tuple[int, int]
    meta_path: str
    audio_path: str


class DallyStableAudioDataset(Dataset):
    """
    Dally dataset:
    - root/audio: <id>.wav
    - root/meta: {audio}_sub-{01..21}_task-{musicFt|justMusic|justFt}.npy
      each .npy has shape (T, 34):
        - first 32 columns: EEG channels
        - column 32: valence
        - column 33: arousal
      sampling frequency: 1000 Hz (so T = 1000 * seconds)
    """
    def __init__(
        self,
        root_dir: str,
        *,
        chunk_dur_s: float = 10.0,
        chunk_overlap_s: float = 2.0,
        # EEG
        eeg_sr: int = 1000,
        use_biot_ch: bool = True,
        biot_dim: int = 18,
        # Audio
        audio_sr: int = 44_100,
        use_classical_only: bool = False,
        # Tasks
        allowed_tasks: Sequence[str] = ("musicFt",),
        # Train / val split
        seed: int = 42,
        split_ratio: float = 0.9,
        split: str = "train",  # or "val"
    ) -> None:
        super().__init__()
        self.root = root_dir
        self.chunk_dur_s = float(chunk_dur_s)
        self.chunk_overlap_s = float(chunk_overlap_s)
        self.eeg_sr = int(eeg_sr)
        self.audio_sr = int(audio_sr)
        self.use_biot_ch = bool(use_biot_ch)
        self.biot_dim = int(biot_dim)
        self.use_classical_only = bool(use_classical_only)
        self.allowed_tasks = tuple(allowed_tasks)

        # Directories
        self.dir_audio = os.path.join(self.root, "audio")
        self.dir_meta = os.path.join(self.root, "meta")
        if not os.path.isdir(self.dir_audio):
            raise FileNotFoundError(f"Audio folder not found: {self.dir_audio}")
        if not os.path.isdir(self.dir_meta):
            raise FileNotFoundError(f"Meta folder not found: {self.dir_meta}")

        # Collect meta files
        meta_paths = sorted(glob.glob(os.path.join(self.dir_meta, "*.npy")))
        if not meta_paths:
            raise FileNotFoundError(f"No .npy meta files found in {self.dir_meta}")

        # Group meta files by audio_id
        meta_by_audio: Dict[str, List[Dict[str, Any]]] = {}
        for mp in meta_paths:
            fname = os.path.basename(mp)  # e.g. "0_sub-01_task-musicFt.npy"
            stem, _ = os.path.splitext(fname)

            # Parse pattern: {audio}_sub-{01..21}_task-{task}
            # crude parsing but robust for the given pattern
            try:
                audio_part, rest = stem.split("_sub-", 1)
                subj_part, task_part = rest.split("_task-", 1)
            except ValueError:
                # Skip unexpected filenames
                continue

            audio_id = audio_part
            subject = subj_part
            task = task_part

            if self.use_classical_only and not _is_classical_audio_id(audio_id):
                continue

            if self.allowed_tasks and task not in self.allowed_tasks:
                continue

            audio_path = os.path.join(self.dir_audio, f"{audio_id}.wav")
            if not os.path.isfile(audio_path):
                # No matching audio; skip this meta file
                continue

            meta_by_audio.setdefault(audio_id, []).append({
                "meta_path": mp,
                "subject": subject,
                "task": task,
                "audio_path": audio_path,
            })

        if not meta_by_audio:
            raise RuntimeError("No usable (audio, meta) pairs found in Dally dataset.")

        # Train / val split by audio_id so the same audio does not appear in both splits
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        rng = np.random.default_rng(seed)
        audio_ids = sorted(meta_by_audio.keys())
        rng.shuffle(audio_ids)
        n_train = max(1, int(round(split_ratio * len(audio_ids))))
        train_ids = set(audio_ids[:n_train])
        val_ids = set(audio_ids[n_train:])
        keep_ids = train_ids if split == "train" else val_ids

        # Build index
        self._meta_cache: Dict[str, np.ndarray] = {}
        self._indices: List[_DallyIdx] = []

        for audio_id in sorted(keep_ids):
            entries = meta_by_audio[audio_id]
            for ent in entries:
                mp = ent["meta_path"]
                subject = ent["subject"]
                task = ent["task"]
                audio_path = ent["audio_path"]

                # Load once to get T for windowing
                arr = np.load(mp)  # (T, 34)
                if arr.ndim != 2 or arr.shape[1] != 34:
                    raise ValueError(f"Expected (T, 34) in {mp}, got {arr.shape}")

                T = arr.shape[0]
                eeg_dur_sec = T / float(self.eeg_sr)
                audio_len_sec = _get_audio_len_sec(audio_path)
                trial_len_sec = min(eeg_dur_sec, audio_len_sec)

                if trial_len_sec <= 0:
                    continue

                chunk_dur_s = self.chunk_dur_s
                overlap_s = self.chunk_overlap_s
                stride_s = max(1e-3, chunk_dur_s - overlap_s)

                starts: List[float] = []
                st = 0.0
                while st + chunk_dur_s <= trial_len_sec:
                    starts.append(st)
                    st += stride_s

                # ensure we have a window that ends exactly at trial_len_sec
                last_possible_start = max(0.0, trial_len_sec - chunk_dur_s)
                if not starts:
                    starts = [0.0]
                elif last_possible_start - starts[-1] > 1e-6:
                    starts.append(last_possible_start)

                for st in starts:
                    eeg_start = int(round(st * self.eeg_sr))
                    eeg_end = eeg_start + int(round(chunk_dur_s * self.eeg_sr))
                    max_eeg_end = int(round(trial_len_sec * self.eeg_sr))
                    if eeg_end > max_eeg_end:
                        eeg_end = max_eeg_end
                    if eeg_end <= eeg_start:
                        continue

                    self._indices.append(_DallyIdx(
                        audio_id=audio_id,
                        subject=subject,
                        task=task,
                        start_sec=st,
                        dur_sec=chunk_dur_s,
                        eeg_slice=(eeg_start, eeg_end),
                        meta_path=mp,
                        audio_path=audio_path,
                    ))

        rng.shuffle(self._indices)

        if not self._indices:
            raise RuntimeError("No windowed indices created for Dally dataset. "
                               "Check chunk_dur_s, overlap, and filtering options.")

    def __len__(self) -> int:
        return len(self._indices)

    def _get_meta(self, meta_path: str) -> np.ndarray:
        if meta_path not in self._meta_cache:
            arr = np.load(meta_path)  # (T, 34)
            if arr.ndim != 2 or arr.shape[1] != 34:
                raise ValueError(f"Expected (T, 34) in {meta_path}, got {arr.shape}")
            self._meta_cache[meta_path] = arr
        return self._meta_cache[meta_path]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self._indices[idx]

        # Load meta and slice
        arr = self._get_meta(x.meta_path)   # (T, 34)
        eeg_all = arr[:, :32].T            # (32, T)
        val_series = arr[:, 32]            # (T,)
        aro_series = arr[:, 33]            # (T,)

        s, e = x.eeg_slice
        eeg = eeg_all[:, s:e]              # (32, Teeg)
        val_win = val_series[s:e]
        aro_win = aro_series[s:e]

        eeg_t = torch.from_numpy(eeg.astype(np.float32))
        if self.use_biot_ch:
            eeg_t = _deap_to_biot_bipolar(eeg_t, self.biot_dim)

        # Audio window aligned in absolute time
        audio_np = _read_audio_segment(
            x.audio_path,
            start_sec=x.start_sec,
            dur_sec=x.dur_sec,
            target_sr=self.audio_sr,
        )
        audio_t = torch.from_numpy(audio_np.astype(np.float32))  # (2, T_audio) or (C, T)

        # Aggregate valence/arousal over the chunk (scalar per item)
        v = float(np.mean(val_win)) if val_win.size > 0 else 0.0
        a = float(np.mean(aro_win)) if aro_win.size > 0 else 0.0

        return {
            "eeg": eeg_t,
            "audio": audio_t,
            "prompt": "",  # keep structure similar to DEAP dataset
            "start_seconds": float(x.start_sec),
            "total_seconds": float(x.dur_sec),
            "valence": v,
            "arousal": a,
            # no dominance / liking for Dally
        }


def create_dally_dataset(
    path: str,
    chunk_dur_s: float = 10.0,
    chunk_overlap_s: float = 2.0,
    use_classical_only: bool = False,
    eeg_sr: int = 1000,
    use_biot_ch: bool = True,
    biot_dim: int = 18,
    audio_sr: int = 44_100,
    allowed_tasks: Sequence[str] = ("musicFt",),
    seed: int = 42,
    split_ratio: float = 0.9,
    split: str = "train",  # or "val"
) -> DallyStableAudioDataset:
    return DallyStableAudioDataset(
        root_dir=path,
        chunk_dur_s=chunk_dur_s,
        chunk_overlap_s=chunk_overlap_s,
        eeg_sr=eeg_sr,
        use_biot_ch=use_biot_ch,
        biot_dim=biot_dim,
        audio_sr=audio_sr,
        use_classical_only=use_classical_only,
        allowed_tasks=allowed_tasks,
        seed=seed,
        split_ratio=split_ratio,
        split=split,
    )


# Quick test

def print_item(ds, idx, should_print: bool):
    sample = ds[idx]
    eeg = sample["eeg"]
    audio = sample["audio"]
    if should_print:
        print(f"item summary for {idx}")
        print(f"  eeg:   shape={tuple(eeg.shape)}, sr={ds.eeg_sr}")
        print(f"  audio: shape={tuple(audio.shape)}, sr={ds.audio_sr}")
        print(f"  start_seconds={sample['start_seconds']} total_seconds={sample['total_seconds']}")
        print(f"  valence={sample['valence']:.4f} arousal={sample['arousal']:.4f}")
        print(f"  prompt=\"{sample['prompt']}\"")


if __name__ == "__main__":
    # Adjust this path to your actual Dally root
    root = "/app/mnt/MusicEEGen/data/dally"

    ds = create_dally_dataset(
        root,
        chunk_dur_s=10.0,
        chunk_overlap_s=2.0,
        use_classical_only=False,
        split="train",
    )

    print(f"Dataset length: {len(ds)}")
    top_n = len(ds)
    for i in range(top_n):
        # Print every 100th item (similar spirit to dataset_deap)
        should_print = (i % 100 == 0)
        print_item(ds, i, should_print)
