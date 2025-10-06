Building a Speech-to-Text Transcriptor with PyTorch & QuartzNet


A Streamlit application that converts speech to text from YouTube links, uploaded audio/video, and documents/pasted text, with optional translation and quality metrics. It supports two ASR engines:

Local QuartzNet (PyTorch) — fast, private, offline with your checkpoint

OpenAI Whisper API — robust cloud transcription for hard/noisy audio

Every run is saved to disk with runtime, engine used, and metrics (WER/CER, plus BLEU when translating).

Table of Contents

What I Built & Why

Why PyTorch

System Design

Features

Folder Layout

Requirements

Setup

Environment Variables (.env)

Run the App

Using the App

Results & Logging

Music / Noisy Audio (Tips)

Troubleshooting

How It Works (Internals)

Future Work

License

What I Built & Why

A practical, Windows-friendly ASR tool that handles real-world inputs (YouTube/audio/video/docs/paste).

A local-first path using PyTorch and a QuartzNet-style model, with a cloud fallback (OpenAI Whisper) for tough cases.

Built-in quality measurement (WER/CER/BLEU) and result logging for reproducibility.

Why PyTorch

The heavy parts of speech recognition—convolutions over long spectrograms, log-Mel extraction, and CTC decoding—are power-hungry. I chose PyTorch to:

Run core logic on tensors (vectorized ops, torch.no_grad()), fast on CPU and accelerated on GPU when available.

Load and run my QuartzNet-style model from a .pt checkpoint with full control over the CTC vocabulary (blank=0; space, apostrophe, a–z).

Prefer torchaudio transforms and automatically fall back to librosa+soundfile when torchaudio wheels are tricky on Windows—so the app still runs anywhere.

System Design

Inputs

YouTube URL → yt-dlp → bestaudio → WAV

Uploaded audio/video → FFmpeg → mono 16k WAV

Documents (TXT/DOCX/PDF/MD/CSV) → text extraction

Pasted text → directly processed (no ASR)

ASR Engines

Local QuartzNet (PyTorch)

Log-Mel features (64 mels), mean/var norm

Greedy CTC decoding (collapse repeats, drop blank=0)

OpenAI Whisper API

File-size guard (~25 MB) with clear error message

Translation

Chunked translation with OpenAI Chat models (default gpt-4o-mini)

Preserves formatting and paragraph breaks

Refusal-resistant prompt + tiny guard for edge cases (e.g., “you” → “tú” not “sí”)

Metrics & Logging

WER/CER via jiwer (when reference text provided)

BLEU via nltk (when translating + reference provided)

Per-run JSON saved to results/ (runtime, engine used, source, metrics, transcript/translation)

Features

ASR: Local QuartzNet (PyTorch) or OpenAI Whisper (toggle in UI)

Inputs: YouTube, audio/video uploads, documents, pasted text

Optional translation to multiple languages

Quality metrics: WER/CER (+ BLEU when translating)

One-click “Music mode (vocal enhance)” (optional) to pre-clean noisy/music clips

Windows-friendly: single environment; FFmpeg auto-detection with .env override

Folder Layout
project/
├─ streamlit_app.py                 # Streamlit entry point (UI + pipeline + logging)
├─ my_utils.py                      # AudioProcessor, log-Mel, greedy CTC, text mapping
├─ test.py                          # QuartzNetTester + WER/CER JSON & summary CSV
├─ checkpoints/                     # Your QuartzNet checkpoint(s)
├─ tools_bin/                       # Optional: ffmpeg.exe, ffprobe.exe
├─ outputs/
│  ├─ audio/                        # Converted audio
│  ├─ docs/                         # Extracted document text
│  └─ translations/<title>/         # source.txt, translated_<lang>.txt
├─ results/                         # Per-run JSON logs (runtime + engine + metrics)
├─ requirements.txt
└─ .env

Requirements

Python 3.9+

FFmpeg + FFprobe available on PATH (or referenced in .env)

One virtual environment (recommended)

The app uses torchaudio if available, otherwise librosa+soundfile automatically.

Setup
Windows (PowerShell)
# 1) Open PowerShell in your project folder
cd "C:\Users\<you>\Desktop\captone final"

# 2) Create & activate a virtual environment
python -m venv speechenv
.\speechenv\Scripts\Activate.ps1

# 3) Upgrade pip (optional)
python -m pip install --upgrade pip

# 4) Install dependencies
pip install -r requirements.txt

macOS / Linux
cd /path/to/project
python3 -m venv speechenv
source ./speechenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Environment Variables (.env)

Create a .env file in the project root:

# If ffmpeg/ffprobe are not on PATH, point to your local copies (Windows example):
FFMPEG_BIN="C:\Users\<you>\Desktop\captone final\tools_bin\ffmpeg.exe"
FFPROBE_BIN="C:\Users\<you>\Desktop\captone final\tools_bin\ffprobe.exe"

# Local QuartzNet checkpoint (update to your file)
QUARTZNET_CHECKPOINT=checkpoints\quartznet_best.pt

# OpenAI (for Whisper + translation)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
# Optional: override translation model
OPENAI_TRANSLATE_MODEL=gpt-4o-mini

Run the App
# Windows
.\speechenv\Scripts\Activate.ps1
streamlit run streamlit_app.py

# macOS/Linux
source ./speechenv/bin/activate
streamlit run streamlit_app.py


Open the local URL shown in the terminal (e.g., http://localhost:8511).

Using the App

Choose an Engine

Local QuartzNet → enter/confirm checkpoint path

OpenAI Whisper → ensure OPENAI_API_KEY is set

Provide Input

YouTube URL, upload audio/video, upload document, or paste text

(Optional) Reference Text

Paste ground truth to compute WER/CER (and BLEU if translating)

(Optional) Translate

Check “Translate the text”, pick a target language

Click Run Pipeline.
Results and logs are written automatically (see below).

Results & Logging

Per run JSON in results/run_YYYYMMDD_HHMMSS.json:

timestamp, runtime_seconds, model_used (Whisper / Local QuartzNet / NeMo QuartzNet / N/A)

source (youtube_url / uploaded_file / document / pasted_text)

audio_path (if any), transcript, translation, reference_text

metrics = {WER, CER, WER_norm, CER_norm, BLEU} (metrics present only if a reference is provided)

Translations in outputs/translations/<title>/:

source.txt

translated_<language>.txt

Testing script outputs (test.py):

Per-item WER JSON in outputs/eval/

Summary CSV with metrics

Music / Noisy Audio (Tips)

ASR struggles with songs (vocals under instruments/reverb). Three options:

1) In-app “Music mode (vocal enhance)” (optional)

Pre-cleans audio (HPF/LPF + denoise + normalize) before ASR. Toggle in the UI.

2) Quick FFmpeg cleanup (manual)
ffmpeg -y -i "song.mp3" `
  -af "highpass=f=120,lowpass=f=4000,afftdn=nf=-20,dynaudnorm" `
  -ac 1 -ar 16000 "clean_song.wav"

3) Separate vocals first (best quality) with Demucs
pip install -U demucs
demucs -n htdemucs -o ".\outputs\demucs" "song.mp3"
ffmpeg -y -i ".\outputs\demucs\htdemucs\song\vocals.wav" `
  -af "highpass=f=120,lowpass=f=4500,afftdn=nf=-15,dynaudnorm" `
  -ac 1 -ar 16000 ".\outputs\demucs\htdemucs\song\vocals_clean16k.wav"


Use the cleaned/isolated vocals WAV in the app.

Troubleshooting

FFmpeg not found / failed
Add FFMPEG_BIN / FFPROBE_BIN in .env or install to PATH. Use the UI’s Check ffmpeg -version to verify.

Whisper “audio too large”
The app checks the ~25 MB API cap and shows a friendly error. Trim the clip or downsample with FFmpeg.

QuartzNet checkpoint mismatch
Ensure your model uses the 29-symbol CTC vocab (blank=0; space, apostrophe, a–z) and compatible feature settings (16 kHz, 64 mels).

torchaudio install issues (Windows)
The app automatically falls back to librosa+soundfile. No action needed.

Dropped PDFs into media uploader
The app restricts types and auto-routes docs to text extraction, but if you see FFmpeg parse errors, use the Document uploader.

How It Works (Internals)

Feature extraction: 16 kHz mono → log-Mel (64 mels, Slaney/HTK) → per-feature mean/var norm

ASR model: QuartzNet-style conv net (CTC).

Decoding: Greedy CTC (collapse repeats, drop blank).

Evaluation: jiwer (WER/CER) and nltk BLEU.

I/O: FFmpeg converts anything to WAV; yt-dlp fetches bestaudio for YouTube.

Logging: structured JSON per run, CSV summaries from test.py.

Future Work

VAD segmentation + batch decoding for long videos

Beam search CTC with LM fusion

CUDA wheels & Demucs GPU path for faster music handling

More translation controls (formal/informal switch, glossary)

License

Personal/educational use. Adapt as needed for course/capstone submission.
