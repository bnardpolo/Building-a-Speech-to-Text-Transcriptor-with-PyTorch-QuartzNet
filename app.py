# streamlit_app.py  — final_api-aligned (QuartzNet or Whisper + optional WER + translate)
import os, re, uuid, subprocess, shutil
from pathlib import Path
from typing import Optional, Tuple, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Local ASR helpers (final_api)
from test import QuartzNetTester, save_result
from my_utils import AudioProcessor

# -------- Optional helpers (graceful if missing) --------
try:
    import yt_dlp
except Exception:
    yt_dlp = None

try:
    import docx
except Exception:
    docx = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# =========================
# Env / Paths / OpenAI
# =========================
load_dotenv()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
AUDIO_DIR = OUT / "audio"
DOC_DIR = OUT / "docs"
TRANS_DIR = OUT / "translations"
for d in (OUT, AUDIO_DIR, DOC_DIR, TRANS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _strip_quotes(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    return x.strip().strip('"').strip("'")

# Prefer PATH ffmpeg/ffprobe; fall back to tools_bin if provided
FFMPEG_BIN  = _strip_quotes(os.getenv("FFMPEG_BIN",  "ffmpeg"))
FFPROBE_BIN = _strip_quotes(os.getenv("FFPROBE_BIN", "ffprobe"))
QUARTZNET_CKPT = os.getenv("QUARTZNET_CHECKPOINT", str(ROOT / "checkpoints" / "quartznet_best.pt"))

def _which_exists(bin_name_or_path: str) -> bool:
    # If absolute/path provided, check directly; otherwise rely on shutil.which
    p = Path(bin_name_or_path)
    if p.is_file():
        return True
    if os.name == "nt":
        # On Windows, allow direct string to exe that is already absolute
        try:
            return Path(bin_name_or_path).exists()
        except Exception:
            pass
    return shutil.which(bin_name_or_path) is not None

def ff_ok() -> Tuple[bool, bool]:
    return _which_exists(FFMPEG_BIN), _which_exists(FFPROBE_BIN)

# OpenAI client (needs OPENAI_API_KEY in env for Whisper)
client = OpenAI()


# =========================
# Utils
# =========================
def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.U).strip()
    text = re.sub(r"\s+", "-", text)
    return (text or "untitled")[:max_len]

def run_ffmpeg(args: List[str]) -> None:
    cmd = [FFMPEG_BIN] + args
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{proc.stderr}")

def to_wav(src: Path, dst_wav: Path, sr: int = 16000) -> None:
    """
    Convert any media to 16 kHz mono PCM16 WAV.
    """
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-nostdin", "-hide_banner", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sr),
        "-acodec", "pcm_s16le",
        str(dst_wav),
    ])

def read_upload_bytes(upload) -> bytes:
    return upload.read() if upload is not None else b""

def read_txt_upload(upload) -> str:
    if upload is None:
        return ""
    data = read_upload_bytes(upload)
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


# =========================
# Inputs → Text helpers
# =========================
def extract_text_from_doc(upload) -> str:
    """TXT/CSV/MD, DOCX, or PDF → text."""
    if upload is None:
        return ""
    name = upload.name.lower()
    if name.endswith((".txt", ".md", ".csv")):
        return read_txt_upload(upload)

    if name.endswith(".docx") and docx is not None:
        tmp = DOC_DIR / f"tmp_{slugify(Path(name).stem)}.docx"
        tmp.write_bytes(read_upload_bytes(upload))
        d = docx.Document(str(tmp))
        return "\n".join(p.text for p in d.paragraphs)

    if name.endswith(".pdf") and PdfReader is not None:
        tmp = DOC_DIR / f"tmp_{slugify(Path(name).stem)}.pdf"
        tmp.write_bytes(read_upload_bytes(upload))
        reader = PdfReader(str(tmp))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return read_txt_upload(upload)

def _guess_ext_for_media(upload) -> str:
    ext = Path(upload.name).suffix.lower()
    if ext: return ext
    mime = getattr(upload, "type", "") or ""
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "video/x-msvideo": ".avi",
        "video/x-ms-wmv": ".wmv",
    }
    return mapping.get(mime, ".bin")


# ---- Friendly size guard for Whisper API ----
# Keep a margin under the ~25 MB API cap.
OPENAI_AUDIO_MAX = 24_000_000

def assert_api_size_ok(audio_path: Path) -> None:
    size = audio_path.stat().st_size
    if size > OPENAI_AUDIO_MAX:
        # Simple, clear guidance (exactly what you asked for)
        raise RuntimeError(
            "This audio is too large for the transcription API. "
            "Please lower the sample rate/bitrate or pick a shorter clip."
        )

def transcribe_audio_wav_openai(wav_path: Path) -> str:
    """OpenAI Whisper API transcription. We guard for oversize."""
    assert_api_size_ok(wav_path)
    with wav_path.open("rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return getattr(result, "text", "") or ""


# =========================
# Shared media loaders
# =========================
def save_media_to_wav(upload, sr: int) -> Tuple[str, Path]:
    """
    Save upload → WAV; returns (title, wav_path)
    """
    ok_ff, ok_fp = ff_ok()
    if not ok_ff or not ok_fp:
        raise FileNotFoundError(
            "FFmpeg/FFprobe not found.\n"
            f"FFMPEG_BIN={FFMPEG_BIN}\n"
            f"FFPROBE_BIN={FFPROBE_BIN}\n"
            "Install ffmpeg in PATH or set .env FFMPEG_BIN/FFPROBE_BIN."
        )
    base = slugify(Path(upload.name).stem) or "upload"
    uniq = uuid.uuid4().hex[:8]
    ext = _guess_ext_for_media(upload)
    src = AUDIO_DIR / "local" / f"{base}_{uniq}{ext}"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(read_upload_bytes(upload))

    wav = AUDIO_DIR / "local" / f"{base}.wav"
    to_wav(src, wav, sr)
    return base, wav

def yt_to_wav(yt_url: str, sr: int) -> Tuple[str, Path]:
    """YouTube → bestaudio → WAV (no transcription here)"""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is required for YouTube downloads (pip install yt-dlp).")
    ok_ff, ok_fp = ff_ok()
    if not ok_ff or not ok_fp:
        raise RuntimeError("FFmpeg/FFprobe not found for YouTube flow.")

    slug = slugify(yt_url)[:40]
    clip_dir = AUDIO_DIR / f"yt_{slug}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(clip_dir / "source.%(ext)s")
    opts = {"format": "bestaudio/best", "outtmpl": outtmpl, "quiet": True, "noprogress": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(yt_url, download=True)
        title = slugify(info.get("title") or "youtube_audio")

    candidates = sorted(clip_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise RuntimeError("Download failed.")
    src = candidates[0]

    wav = clip_dir / f"{title}.wav"
    to_wav(src, wav, sr)
    return title, wav


# =========================
# Translation via OpenAI (full text)
# =========================
LANGS = [
    "Spanish", "French", "German", "Portuguese", "Italian",
    "Arabic", "Chinese (Simplified)", "Chinese (Traditional)",
    "Japanese", "Korean", "Hindi", "Russian", "English"
]

SYSTEM_TRANSLATE_TEMPLATE = (
    "You are a professional translator. "
    "Translate the user's text into {target_lang}. "
    "Preserve meaning, names, numbers, tone, and paragraph breaks. "
    "Do not summarize; translate everything verbatim except code blocks. "
    "If code blocks appear (``` … ```), leave them unchanged."
)

def translate_big_text_openai(text: str, target_lang: str, model: str = "gpt-4o-mini") -> str:
    if not text.strip():
        return ""
    CHUNK_SIZE = 6000
    chunks = []
    cur = 0
    while cur < len(text):
        end = min(cur + CHUNK_SIZE, len(text))
        segment = text[cur:end]
        if end < len(text):
            last_nl = segment.rfind("\n")
            if last_nl > 1000:
                end = cur + last_nl
                segment = text[cur:end]
        chunks.append(segment)
        cur = end

    pieces = []
    for i, chunk in enumerate(chunks, 1):
        with st.spinner(f"Translating chunk {i}/{len(chunks)}…"):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": SYSTEM_TRANSLATE_TEMPLATE.format(target_lang=target_lang)},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.2,
            )
            pieces.append(resp.choices[0].message.content.strip())
    return "\n".join(pieces).strip()


# =========================
# UI
# =========================
st.set_page_config(page_title="ASR + Translate (QuartzNet or Whisper)", layout="wide")
st.title("Speech → Text → (Optional) Translation")

left, main = st.columns([0.30, 0.70])

with left:
    st.subheader("Engine & Settings")
    engine = st.radio("ASR Engine", ["Local QuartzNet", "OpenAI Whisper"], index=0)
    sr = st.number_input("Sample rate for audio (Hz)", value=16000, step=1000)
    if engine == "Local QuartzNet":
        ckpt = st.text_input("QuartzNet checkpoint", value=QUARTZNET_CKPT)
    else:
        ckpt = ""

    st.markdown("---")
    st.subheader("Optional WER/CER")
    ref_text = st.text_area("Reference text (paste ground truth to compute WER/CER)", height=120)

    st.markdown("---")
    st.subheader("FFmpeg status")
    ok_ff, ok_fp = ff_ok()
    st.caption(f"FFMPEG_BIN = `{FFMPEG_BIN}`")
    st.caption(f"FFPROBE_BIN = `{FFPROBE_BIN}`")
    st.success("ffmpeg found: Yes" if ok_ff else "ffmpeg found: No")
    st.success("ffprobe found: Yes" if ok_fp else "ffprobe found: No")
    if st.button("Check `ffmpeg -version`"):
        if ok_ff:
            out = subprocess.run([FFMPEG_BIN, "-version"], stdout=subprocess.PIPE, text=True)
            st.code(out.stdout[:1000] + ("..." if len(out.stdout) > 1000 else ""))
        else:
            st.warning("ffmpeg not found")

with main:
    st.markdown("#### Source options")
    yt_url = st.text_input("YouTube URL (optional)")
    up_media = st.file_uploader("Upload audio or video (optional)", type=None)
    up_doc = st.file_uploader(
        "Upload a document (.txt, .docx, .pdf, .md, .csv) (optional)",
        type=["txt", "docx", "pdf", "md", "csv"]
    )
    pasted = st.text_area("Or paste text here…", height=160)

    st.markdown("#### Output options")
    do_translate = st.checkbox("Translate the text", value=False)
    target_lang = st.selectbox("Translate to", LANGS, index=0, disabled=not do_translate)

    if st.button("Run Pipeline", use_container_width=True):
        try:
            # Decide the source of TEXT:
            # Priority: YouTube/audio/video → transcribe; else document/paste → direct text
            title = "session"
            text_source = ""

            if yt_url.strip():
                # Download → WAV
                title, wav_path = yt_to_wav(yt_url.strip(), sr)
            elif up_media is not None:
                # Upload → WAV
                title, wav_path = save_media_to_wav(up_media, sr)
            else:
                # Non-audio path: document or pasted text
                if up_doc is not None:
                    title = slugify(Path(up_doc.name).stem)
                    text_source = extract_text_from_doc(up_doc)
                    if not text_source.strip():
                        st.error("Could not extract any text from the document.")
                        st.stop()
                    st.success(f"Document text extracted: {title}")
                else:
                    if not pasted.strip():
                        st.error("No text to process. Enter a YouTube URL, upload audio/video/document, or paste text.")
                        st.stop()
                    text_source = pasted.strip()
                    title = "pasted-text"

            # If we have a WAV path, do ASR
            if 'wav_path' in locals():
                if engine == "OpenAI Whisper":
                    text_source = transcribe_audio_wav_openai(wav_path)
                    model_name = "whisper-1"
                else:
                    # Local QuartzNet
                    if not ckpt or not Path(ckpt).exists():
                        st.error("QuartzNet checkpoint not found. Provide a valid path.")
                        st.stop()
                    proc = AudioProcessor(sample_rate=sr)
                    tester = QuartzNetTester(ckpt, proc)
                    text_source = tester.transcribe(str(wav_path))
                    model_name = "QuartzNet"

                if not text_source.strip():
                    st.error("Transcription returned no text.")
                    st.stop()
                st.success(f"Transcribed: {title}")

                # Optional WER/CER (only if reference provided)
                if ref_text.strip():
                    out_json, metrics = save_result(str(wav_path), ref_text, text_source, model_name=model_name)
                    st.info(f"Saved metrics JSON: {out_json}")
                    st.write({"WER_raw": metrics["WER_raw"], "CER_raw": metrics["CER_raw"],
                              "WER_norm": metrics["WER_norm"], "CER_norm": metrics["CER_norm"]})

            # Translate (optional)
            if do_translate:
                with st.spinner(f"Translating to {target_lang}…"):
                    translated = translate_big_text_openai(text_source, target_lang)
                # Save outputs
                session_dir = TRANS_DIR / slugify(title)
                session_dir.mkdir(parents=True, exist_ok=True)
                src_path = session_dir / "source.txt"
                out_path = session_dir / f"translated_{slugify(target_lang)}.txt"
                src_path.write_text(text_source, encoding="utf-8")
                out_path.write_text(translated, encoding="utf-8")

                st.success("Translation complete.")
                st.subheader("Original")
                st.text_area("Original text", text_source, height=220)
                st.subheader(f"Translated ({target_lang})")
                st.text_area("Translated text", translated, height=260)

                st.download_button(
                    "Download translation (.txt)",
                    data=translated.encode("utf-8"),
                    file_name=out_path.name,
                    mime="text/plain",
                )
                st.info(f"Saved files in: {session_dir}")
            else:
                st.subheader("Transcript / Text")
                st.text_area("Text", text_source, height=260)

        except Exception as e:
            # Single place to surface friendly message for big audio (Whisper path)
            emsg = str(e)
            if ("too large" in emsg.lower()) or ("maximum content size" in emsg.lower()) or ("413" in emsg):
                st.error(
                    "This audio is too large for the transcription API. "
                    "Please lower the sample rate/bitrate or pick a shorter clip."
                )
            else:
                st.error(emsg)
