peech Recognition & Translation App

A Streamlit application for speech-to-text and translation that accepts YouTube links, uploaded audio/video, documents, and pasted text. It supports two ASR engines:

Local QuartzNet (PyTorch) — fast, private, offline with your checkpoint

OpenAI Whisper API — robust cloud transcription for difficult/noisy audio

Each run saves a JSON log with runtime, engine used, and quality metrics (WER, CER, and BLEU when a reference is provided).

1) Overview

Windows-friendly UI with deterministic audio conversion to mono 16 kHz

Modular inputs (YouTube, media files, documents, pasted text)

Optional translation to multiple languages

Optional metrics: WER, CER, BLEU + structured per-run logs

Clear fallback plan: if Local QuartzNet struggles, switch to OpenAI Whisper

2) Why PyTorch

I used PyTorch to keep the local pipeline fast and under full control:

Feature extraction: log-Mel spectrograms (64 bins, Slaney/HTK), mean/var normalization

Modeling: QuartzNet-style 1D conv network with CTC decoding (blank=0; space, apostrophe, a–z)

Performance: vectorized ops with torch.no_grad() run well on CPU and are GPU-ready

Portability: prefers torchaudio but automatically falls back to librosa + soundfile on Windows when needed

3) System Design

Inputs

YouTube → yt-dlp → bestaudio → WAV

Audio/Video → FFmpeg → mono 16 kHz WAV

Documents (TXT/DOCX/PDF/MD/CSV) → text extraction

Pasted text → used directly

ASR Engines

Local QuartzNet (PyTorch, greedy CTC)

OpenAI Whisper API with file-size guard

Translation

OpenAI chat model (default gpt-4o-mini), chunked, formatting preserved

Metrics & Logging

WER/CER via jiwer

BLEU via nltk

Per-run JSON logs (runtime, engine, sources, metrics, transcript/translation)

4) Folder Layout
project/
├─ streamlit_app.py              # UI + pipeline + logging
├─ my_utils.py                   # AudioProcessor, log-Mel, greedy CTC, text mapping
├─ test.py                       # QuartzNetTester + WER/CER JSON & summary CSV
├─ checkpoints/                  # Your QuartzNet checkpoint(s)
├─ tools_bin/                    # Optional: ffmpeg.exe, ffprobe.exe (Windows)
├─ outputs/
│  ├─ audio/                     # Converted audio
│  ├─ docs/                      # Extracted document text
│  └─ translations/<title>/      # source.txt, translated_<lang>.txt
├─ results/                      # Per-run JSON logs (runtime + engine + metrics)
├─ requirements.txt
└─ .env

5) Requirements

Python 3.9+

FFmpeg and FFprobe available on PATH, or pointed to in .env

One virtual environment recommended

torchaudio optional (automatic fallback to librosa + soundfile is built in)