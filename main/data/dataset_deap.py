
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


def _default_prompt(deap_block: Dict[str, Any], ratings: Optional[np.ndarray]) -> str:
    tag = str(deap_block.get('Lastfm_tag', '') or '').strip()
    artist = str(deap_block.get('Artist', '') or '').strip()
    title = str(deap_block.get('Title', '') or '').strip()
    bits = []
    if artist or title or tag:
        bits.append(f"Tag: {tag} - Artist: {artist} — Title: {title}".strip(' —'))
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
        chunk_dur_s: float = 10.0, # Windowing
        chunk_overlap_s: float = 2.0,
        # EEG
        eeg_sr: int = 128,
        include_peripheral: bool = False,
        drop_baseline_3s: bool = True,
        use_biot_ch: bool=True,
        biot_dim: int=18,
        # Audio
        audio_sr: int = 44_100,
        seed: int = 42,
        use_prompt: bool = True,
        preprocess_audio_path: str = None,
        # Train Test split
        split_ratio: float = 0.9,
        split: str = 'train' # or val
    ) -> None:
        super().__init__()
        self.root = root_dir
        self.chunk_dur_s = float(chunk_dur_s)
        self.chunk_overlap_s = float(chunk_overlap_s)
        self.eeg_sr = int(eeg_sr)
        self.include_peripheral = bool(include_peripheral)
        self.drop_baseline_3s = bool(drop_baseline_3s)
        self.audio_sr = int(audio_sr)
        self.use_prompt = use_prompt
        self.use_biot_ch = use_biot_ch
        self.biot_dim = biot_dim

        # Directories
        if preprocess_audio_path is not None:
            self.dir_audio = preprocess_audio_path
            self.use_preprocessed_audio = True
        else:
            self.dir_audio = os.path.join(self.root, 'audio')
            self.use_preprocessed_audio = False
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
            a_path = f'{stem}_cropped.wav'
           
            deap_block = meta.get('deap', {}) or {}
            
            hstart = 0 if self.use_preprocessed_audio else float(deap_block.get('Highlight_start', 0.0) or 0.0)
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
            s_code = os.path.splitext(os.path.basename(dat_path))[0] # 's01'
            subj = _load_subject_dat(dat_path) # dict of two tensors
            data = subj['data']    # (40, 40, 8064)
            labels = subj['labels']  # (40, 4)

            total_samples = int(data.shape[-1])
            # DEAP: 3s baseline + 60 s trial @ 128 Hz → 3*128 + 60*128 = 8064
            if self.drop_baseline_3s and total_samples >= int(63 * self.eeg_sr):
                trial_offset_sec = 3.0
            else:
                trial_offset_sec = 0.0

            trial_len_sec = 60.0          # full trial length
            chunk_dur_s   = self.chunk_dur_s  
            overlap_s     = self.chunk_overlap_s

            stride_s = max(1e-3, chunk_dur_s - overlap_s) 

            starts = []
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

            # For each available experiment id, map to trial index
            for exp_id, blk in self.experiments.items():
                trial = exp_id - 1 # exp is 1-indexed; trial is 0-indexed
                if not (0 <= trial < data.shape[0]):
                    continue 

                audio_path = blk['audio']
                audio_len_sec = _get_audio_len_sec(audio_path)

                for st in starts:
                    eeg_start = int((trial_offset_sec + st) * self.eeg_sr)
                    eeg_end = eeg_start + int(self.chunk_dur_s * self.eeg_sr)
                    max_end = int(trial_offset_sec * self.eeg_sr) + int(trial_len_sec * self.eeg_sr)
                    if eeg_end > max_end: # handle precision error
                        eeg_end = max_end

                    a_start = float(blk['hstart']) + st
                    # skip if audio is too short for this window, inaccurate label
                    if a_start + self.chunk_dur_s > audio_len_sec:
                        continue

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
        if self.use_biot_ch:
            eeg_t = _deap_to_biot_bipolar(eeg_t, self.biot_dim)

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
            prompt = _default_prompt(deap_block, ratings)
        else:
            prompt = ''

        return {
            'eeg': eeg_t,
            'audio': audio_t,
            'prompt': prompt,
            # TODO check how is start and total seconds are used!!
            'start_seconds': float(0),
            'total_seconds': float(self.chunk_dur_s),
        }



def create_deap_dataset(
    path: str,
    chunk_dur_s: float = 47.55446713,
    chunk_overlap_s: float = 2.0,
    eeg_sr: int = 128,
    use_biot_ch: bool = True,
    biot_dim: int=18,
    include_peripheral: bool = False,
    drop_baseline_3s: bool = True,
    audio_sr: int = 44100,
    seed: int = 42,
    use_prompt: bool = True,
    split_ratio: float = 0.9,
    split: str = 'train' # or val
) -> DEAPStableAudioDataset:
    return DEAPStableAudioDataset(
        path,
        chunk_dur_s=chunk_dur_s,
        eeg_sr=eeg_sr,
        use_biot_ch=use_biot_ch,
        biot_dim=biot_dim,
        include_peripheral=include_peripheral,
        drop_baseline_3s=drop_baseline_3s,
        audio_sr=audio_sr,
        seed=seed,
        use_prompt=use_prompt,
        split_ratio=split_ratio,
        split=split
    )

   
# Quick test

def print_item(ds, id, shoud_print):
    sample = ds[id]
    eeg = sample['eeg']
    audio = sample['audio']
    if shoud_print:
        print(f"item summary for {id}")
        print(f"  eeg:   shape={tuple(eeg.shape)}, sr={ds.eeg_sr}")
        print(f"  audio: shape={tuple(audio.shape)}, sr={ds.audio_sr}")
        print(f"  start_seconds={sample['start_seconds']} total_seconds={sample['total_seconds']}")
        print(f"  prompt=\"{sample['prompt']}\"")


if __name__ == '__main__':
    root = '/app/mnt/MusicEEGen/data/deap/deap-dataset'

    ds = create_deap_dataset(root, split='train')

    print(f"Dataset length: {len(ds)}")
    top_n = len(ds)
    for i in range(top_n):
        if top_n % 100 == 0:
            print_item(ds, i, True)
        else:
            print_item(ds, i, False)
    
