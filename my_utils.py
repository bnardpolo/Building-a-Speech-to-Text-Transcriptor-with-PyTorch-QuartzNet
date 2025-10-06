# Audio processing utilities for QuartzNet ASR


from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List

import torch
import numpy as np

# -------------------------
# Optional backends
# -------------------------
HAVE_TORCHAUDIO = False
try:
    import torchaudio
    HAVE_TORCHAUDIO = True
except Exception:
    # Fallback stack for Windows if torchaudio wheels are tricky
    import librosa
    import soundfile as sf  # pip install soundfile librosa


# -------------------------
# Text mapping for CTC
# -------------------------
# NOTE: CTCLoss(blank=0) in your train/test code, so:
# 0 = <blank>, 1 = space, 2 = apostrophe, 3..28 = a..z
IDX_TO_CHAR = ["<blank>", " ", "'"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
CHAR_TO_IDX = {ch: i for i, ch in enumerate(IDX_TO_CHAR)}

def normalize_text(s: str) -> str:
    # Lowercase and keep only a–z, space, apostrophe (aligns with the 29-char set)
    s = s.lower()
    out = []
    for ch in s:
        if ch in " '":
            out.append(ch)
        elif "a" <= ch <= "z":
            out.append(ch)
        # else drop punctuation, digits, etc.
    # collapse multiple spaces
    return " ".join("".join(out).split())

def text_to_indices(s: str) -> List[int]:
    s = normalize_text(s)
    return [CHAR_TO_IDX[ch] for ch in s if ch in CHAR_TO_IDX]

def indices_to_text(idxs: List[int]) -> str:
    chars = []
    for i in idxs:
        if i <= 0 or i >= len(IDX_TO_CHAR):
            # skip <blank> or out-of-range
            continue
        chars.append(IDX_TO_CHAR[i])
    return "".join(chars)


# -------------------------
# Audio processing
# -------------------------
@dataclass
class MelParams:
    sample_rate: int = 16000
    n_fft: int = 512            # ~32ms at 16k
    win_length: int = 400       # 25ms window
    hop_length: int = 160       # 10ms hop
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float | None = None  # None -> sr/2
    power: float = 2.0          # power spectrum
    eps: float = 1e-10          # numerical stability


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, mel_params: MelParams | None = None):
        self.sample_rate = sample_rate
        self.mel = mel_params or MelParams(sample_rate=sample_rate)

        if HAVE_TORCHAUDIO:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.mel.sample_rate,
                n_fft=self.mel.n_fft,
                win_length=self.mel.win_length,
                hop_length=self.mel.hop_length,
                f_min=self.mel.f_min,
                f_max=self.mel.f_max,
                n_mels=self.mel.n_mels,
                power=self.mel.power,
                center=True,
                norm="slaney",
                mel_scale="htk",
            )
            # We’ll convert to log with clamp later for numerical stability.

    # -------- I/O --------
    def load_audio(self, path: str) -> torch.Tensor:
        """
        Returns a 1D float32 tensor (T,) at self.sample_rate.
        """
        if HAVE_TORCHAUDIO:
            wav, sr = torchaudio.load(path)  # (C, T)
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)  # mono
            else:
                wav = wav.reshape(1, -1)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            return wav.squeeze(0).to(torch.float32)
        else:
            y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            # Ensure float32 tensor
            return torch.from_numpy(y.astype(np.float32))

    # -------- Features --------
    def audio_to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: waveform (T,) float32
        Output: log-Mel spectrogram (n_mels, time) float32
        """
        if waveform.dim() != 1:
            waveform = waveform.view(-1)

        if HAVE_TORCHAUDIO:
            # expects (batch, time) or (time,)
            spec = self._mel_transform(waveform.unsqueeze(0))  # (1, n_mels, time)
            spec = spec.squeeze(0)  # (n_mels, time)
            # Log compression: log(1 + mels)
            spec = torch.log(spec.clamp_min(self.mel.eps))
        else:
            mels = librosa.feature.melspectrogram(
                y=waveform.numpy(),
                sr=self.mel.sample_rate,
                n_fft=self.mel.n_fft,
                hop_length=self.mel.hop_length,
                win_length=self.mel.win_length,
                n_mels=self.mel.n_mels,
                fmin=self.mel.f_min,
                fmax=self.mel.f_max,
                power=self.mel.power,
                center=True,
                htk=True,
                norm="slaney",
            )
            spec = np.log(np.clip(mels, self.mel.eps, None)).astype(np.float32)
            spec = torch.from_numpy(spec)

        # Optional mean/var normalization (helpful for stability)
        spec = spec - spec.mean(dim=1, keepdim=True)
        std = spec.std(dim=1, keepdim=True).clamp_min(1e-5)
        spec = spec / std
        return spec

    # -------- Decoding --------
    def greedy_ctc_decode(self, logits: torch.Tensor) -> List[int]:
        """
        Greedy decode from frame-wise logits.
        logits: (C, T) or (T, C). We’ll handle both.
        Returns a list of label indices after collapsing repeats and removing blank (0).
        """
        if logits.dim() != 2:
            raise ValueError("logits must be 2D (C,T) or (T,C)")
        C, T = logits.shape[0], logits.shape[1]
        # If it looks like (T,C), transpose
        if C < T and C != len(IDX_TO_CHAR):
            # likely (T,C)
            logits = logits.transpose(0, 1)  # (C,T)

        # Argmax per frame
        path = torch.argmax(logits, dim=0).tolist()  # length T
        # Collapse repeats, remove blanks
        collapsed = []
        prev = None
        for p in path:
            if p == 0:            # blank
                prev = p
                continue
            if p != prev:
                collapsed.append(p)
            prev = p
        return collapsed

    def decode_predictions(self, logits: torch.Tensor) -> str:
        """
        logits: (C,T) or (T,C). Returns decoded string.
        """
        ids = self.greedy_ctc_decode(logits)
        return indices_to_text(ids)

    # -------- Optional helpers used during training --------
    def encode_text(self, s: str) -> torch.Tensor:
        """
        Normalize and convert text to a 1D LongTensor of label indices
        (without CTC blanks).
        """
        ids = text_to_indices(s)
        return torch.tensor(ids, dtype=torch.long)

    def pad_collate(self, batch: List[dict]):
        """
        Example collate_fn for DataLoader. Expects each item:
        {
          'mel_spec': Tensor[n_mels, T],
          'text': str
        }
        Returns:
          mel_specs: Tensor[B, n_mels, T_max]
          text_targets: 1D Tensor[sum(target_lengths)]
          target_lengths: 1D Tensor[B]
        """
        # Pad spectrograms
        lengths = [b["mel_spec"].shape[1] for b in batch]
        T_max = max(lengths)
        n_mels = batch[0]["mel_spec"].shape[0]
        B = len(batch)
        mel_specs = torch.zeros((B, n_mels, T_max), dtype=torch.float32)
        for i, b in enumerate(batch):
            t = b["mel_spec"].shape[1]
            mel_specs[i, :, :t] = b["mel_spec"]

        # Flatten text targets for CTC
        targets_list = [self.encode_text(b["text"]) for b in batch]
        target_lengths = torch.tensor([t.numel() for t in targets_list], dtype=torch.long)
        text_targets = torch.cat(targets_list) if targets_list else torch.zeros(0, dtype=torch.long)

        return {
            "mel_spec": mel_specs,
            "text": text_targets,
            "target_lengths": target_lengths,
        }
