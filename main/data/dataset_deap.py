#!/usr/bin/env python3
"""
DEAP → Stable Audio Dataset (EEG–Audio alignment)

This module exposes:

    def create_deap_dataset(root_dir: str, **kwargs) -> torch.utils.data.Dataset

It builds a map-style Dataset that:
- Loads all DEAP subject files from:  <root_dir>/data_preprocessed_python/sXX.dat
- Loads available audio clips + metadata JSONs from: <root_dir>/audio/<id>.(json|m4a|webm|opus|mp3|wav)
- Uses the JSON's `deap.Highlight_start` (seconds) to align audio with the 60 s DEAP trial
- Windows each 60 s trial into chunk(s) (default 47.554 s) with optional hop
- Time-aligns EEG windows (after optional 3 s baseline drop) with the audio windows
- Optionally resamples audio to Stable Audio's common SR (default 44_100)

get_item returns a dict with at least these keys:
  - 'eeg':   FloatTensor (C_eeg, T_eeg)
  - 'audio': FloatTensor (2, T_audio)
  - 'prompt': str
  - 'start_seconds': float  # relative to the 60s trial window
  - 'total_seconds': float  # usually 60.0

Notes
-----
• Trial index mapping: by default, we assume `experiment_id` (1-indexed as in JSON filenames)
  maps to DEAP trial index `experiment_id - 1`. If your experiment ordering differs,
  pass a custom `expid_to_trial` mapping dict via the factory.
• DEAP .dat structure (preprocessed): data shape is (40 trials, 40 channels, 8064 samples),
  labels shape is (40, 4). The 8064 samples typically cover 3 s baseline + 60 s trial @ 128 Hz.

"""

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


def _load_subject_dat(dat_path: str) -> Dict[str, np.ndarray]:
    """
    Take in preprocessed eeg .dat file
    Return dictionary of two tensors
    data: (40, 40, 8064)
    labels: (40, 4): valence, arousal, dominance, liking
    """
    import pickle
    with open(dat_path, 'rb') as f:
        obj = pickle.load(f, encoding='latin1') if hasattr(pickle, 'load') else pickle.load(f)
    if isinstance(obj, dict) and 'data' in obj and 'labels' in obj:
        return obj
    raise ValueError(f"Unrecognized DEAP .dat structure at {dat_path}")


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

def _read_audio_segment(path: str, start_sec: float, dur_sec: float, target_sr: int,
                        force_stereo: bool = True) -> np.ndarray:
    """
    Load [start_sec, start_sec + dur_sec) from a WAV file using torchaudio,
    resample to target_sr, return (C, T) float32. Raises on out-of-bounds or short audio.
    """
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

    return window.numpy().astype(np.float32, copy=False)


def _default_prompt(deap_block: Dict[str, Any], ratings: Optional[np.ndarray]) -> str:
    tag = str(deap_block.get('Lastfm_tag', '') or '').strip()
    artist = str(deap_block.get('Artist', '') or '').strip()
    title = str(deap_block.get('Title', '') or '').strip()
    bits = []
    if artist or title:
        bits.append(f"{artist} — {title}".strip(' —'))
    if tag:
        bits.append(f"tag: {tag}")
    if ratings is not None and ratings.size == 4:
        v, a, d, l = [float(x) for x in ratings]
        bits.append(f"valence {v:.2f}/9 arousal {a:.2f}/9 dominance {d:.2f}/9 liking {l:.2f}/9")
    return "; ".join([b for b in bits if b])


@dataclass
class _Idx:
    subject: str        # 's01'
    trial: int          # 0..39
    experiment_id: int  # 1-based matched the xls
    start_sec: float    # within the 60 s trial (window start)
    dur_sec: float
    eeg_slice: Tuple[int, int]
    highlight_start: float
    audio_json: str
    audio_path: str


class DEAPStableAudioDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        *,
        chunk_dur_s: float = 47.55446713, # Windowing
        # EEG
        eeg_sr: int = 128,
        include_peripheral: bool = False,
        drop_baseline_3s: bool = True,
        # Audio
        audio_sr: int = 44_100,
        seed: int = 42,
        use_prompt: bool = True,
        # Train Test split
        split_ratio: float = 0.9,
        split: str = 'train' # or val
    ) -> None:
        super().__init__()
        self.root = root_dir
        self.chunk_dur_s = float(chunk_dur_s)
        self.eeg_sr = int(eeg_sr)
        self.include_peripheral = bool(include_peripheral)
        self.drop_baseline_3s = bool(drop_baseline_3s)
        self.audio_sr = int(audio_sr)
        self.use_prompt = use_prompt

        # Directories
        self.dir_audio = os.path.join(self.root, 'audio')
        self.dir_dat = os.path.join(self.root, 'data_preprocessed_python')
        if not os.path.isdir(self.dir_audio):
            raise FileNotFoundError(f"Audio folder not found: {self.dir_audio}")
        if not os.path.isdir(self.dir_dat):
            raise FileNotFoundError(f"DEAP preprocessed folder not found: {self.dir_dat}")

        # Find available audio JSONs (these define the set of usable experiments)
        json_paths = sorted(glob.glob(os.path.join(self.dir_audio, '*.json')))
        if not json_paths:
            raise FileNotFoundError(f"No audio JSONs found in {self.dir_audio}")

        # Build experiment_id → {json_path, audio_path, deap(meta), highlight_start}
        self.experiments: Dict[int, Dict[str, Any]] = {}
        for jp in json_paths:
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                continue
            try:
                exp_id = int(str(meta.get('experiment_id')).strip())
            except Exception:
                continue
            stem = os.path.splitext(jp)[0]
            a_path = f'{stem}.wav'
           
            deap_block = meta.get('deap', {}) or {}
            hstart = float(deap_block.get('Highlight_start', 0) or 0)
            self.experiments[exp_id] = {
                'json': jp,
                'audio': a_path,
                'deap': deap_block,
                'hstart': hstart,
            }

        # Train test split based on experiment (audio)
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        exp_ids = sorted(self.experiments.keys())
        if not exp_ids:
            raise RuntimeError("No usable experiments (json+wav) found.")

        rng = np.random.default_rng(seed)     
        rng.shuffle(exp_ids)
        n_train = max(1, int(round(split_ratio * len(exp_ids)))) 
        train_ids = set(exp_ids[:n_train])
        val_ids = set(exp_ids[n_train:])
        keep_ids = train_ids if split == "train" else val_ids
        self.experiments = {eid: info for eid, info in self.experiments.items() if eid in keep_ids}

        if not self.experiments:
            raise RuntimeError("No usable experiments (audio+json) found.")

        # Subject files
        all_dat = sorted(glob.glob(os.path.join(self.dir_dat, 's*.dat')))
        if not all_dat:
            raise FileNotFoundError(f"No subject .dat files in {self.dir_dat}")

        # Build index list
        self._subject_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._indices: List[_Idx] = []
        rng = np.random.default_rng(seed)

        for dat_path in all_dat:
            s_code = os.path.basename(dat_path)  # 's01'
            subj = _load_subject_dat(dat_path) # dict of two tensors
            data = subj['data']    # (40, 40, 8064)
            labels = subj['labels']  # (40, 4)

            total_samples = int(data.shape[-1])
            # DEAP: 3s baseline + 60 s trial @ 128 Hz → 3*128 + 60*128 = 8064
            if self.drop_baseline_3s and total_samples >= int(63 * self.eeg_sr):
                trial_offset_sec = 3.0
            else:
                trial_offset_sec = 0.0

            # 2 windows within 60 s
            trial_len_sec = 60.0
            starts = [0.0, max(0.0, trial_len_sec - chunk)]

            # For each available experiment id, map to trial index
            for exp_id, blk in self.experiments.items():
                trial = exp_id - 1 # exp is 1-indexed; trial is 0-indexed
                if not (0 <= trial < data.shape[0]):
                    continue 

                for st in starts:
                    eeg_start = int((trial_offset_sec + st) * self.eeg_sr)
                    eeg_end = eeg_start + int(self.chunk_dur_s * self.eeg_sr)
                    max_end = int(trial_offset_sec * self.eeg_sr) + int(trial_len_sec * self.eeg_sr)
                    if eeg_end > max_end: # handle precision error
                        eeg_end = max_end

                    self._indices.append(_Idx(
                        subject=s_code, # s01 
                        trial=trial, 
                        experiment_id=exp_id,
                        start_sec=st,
                        dur_sec=self.chunk_dur_s,
                        eeg_slice=(eeg_start, eeg_end),
                        highlight_start=float(blk['hstart']),
                        audio_json=blk['json'], # json path
                        audio_path=blk['audio'], # audio path
                    ))

        rng.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def _get_subject(self, s_code: str) -> Dict[str, np.ndarray]:
        if s_code not in self._subject_cache:
            path = os.path.join(self.dir_dat, f"{s_code}.dat")
            self._subject_cache[s_code] = _load_subject_dat(path)
        return self._subject_cache[s_code]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self._indices[idx]
        subj = self._get_subject(x.subject)
        data = subj['data']    # (40, 40, T)
        labels = subj['labels']  # (40, 4)

        # EEG channels: first 32 are EEG, remaining 8 are peripheral
        ch = 40 if self.include_peripheral else 32
        trial_arr = data[x.trial][:ch]  # (C, T)
        s, e = x.eeg_slice
        eeg = trial_arr[:, s:e]
        eeg_t = torch.from_numpy(eeg.astype(np.float32))  # (C, T_eeg)

        # Audio window aligned with EEG window: start at highlight_start + start_sec
        a_start = x.highlight_start + x.start_sec
        audio_np = _read_audio_segment(x.audio_path, start_sec=a_start, dur_sec=x.dur_sec, target_sr=self.audio_sr)
        audio_t = torch.from_numpy(audio_np.astype(np.float32))  # (2, T_audio)

        # Prompt: lastfm tag + artist/title + subject-specific labels for this trial
        ratings = labels[x.trial].astype(np.float32) if labels is not None else None
        with open(x.audio_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        deap_block = meta.get('deap', {}) or {}
        if self.use_prompt:
            prompt = self.prompt_fn(deap_block, ratings)
        else:
            prompt = ''

        return {
            'eeg': eeg_t,
            'audio': audio_t,
            'prompt': prompt,
            # TODO check how is start and total seconds are used!!
            'start_seconds': float(x.start_sec),
            'total_seconds': float(self.chunk_dur_s),
        }



def create_deap_dataset(
    root_dir: str,
    *,
    chunk_dur_s: float = 47.55446713,
    eeg_sr: int = 128,
    include_peripheral: bool = False,
    drop_baseline_3s: bool = True,
    audio_sr: int = 44100,
    seed: int = 42,
    use_prompt: bool = True,
    split_ratio: float = 0.9,
    split: str = 'train' # or val
) -> DEAPStableAudioDataset:
    return DEAPStableAudioDataset(
        root_dir,
        chunk_dur_s=chunk_dur_s,
        eeg_sr=eeg_sr,
        include_peripheral=include_peripheral,
        drop_baseline_3s=drop_baseline_3s,
        audio_sr=audio_sr,
        seed=seed,
        use_prompt=use_prompt,
        split_ratio=split_ratio,
        split=split
    )

   
# Quick test

if __name__ == '__main__':
    root = '/app/mnt/MusicEEGen/data/deap/deap-dataset'

    ds = create_deap_dataset(root, split='train')

    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        eeg = sample['eeg']
        audio = sample['audio']
        print("First item summary →")
        print(f"  subject/trial/exp: {sample['subject']}/{sample['trial']}/{sample['experiment_id']}")
        print(f"  eeg:   shape={tuple(eeg.shape)}, sr={ds.eeg_sr}")
        print(f"  audio: shape={tuple(audio.shape)}, sr={ds.audio_sr}")
        print(f"  start_seconds={sample['start_seconds']} total_seconds={sample['total_seconds']}")
        print(f"  prompt=\"{sample['prompt']}\"")
