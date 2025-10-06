# test.py — Local QuartzNet testing + WER/CER (supports .pt/.pth and .nemo)
import os
import time
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

from my_utils import AudioProcessor
from models import QuartzNet  # used for .pt/.pth checkpoints

# ----- Optional NeMo (only needed for .nemo checkpoints) -----
try:
    from nemo.collections.asr.models import ASRModel
    _NEMO_OK = True
except Exception:
    _NEMO_OK = False


# ===============================
# Utility: flexible ckpt handling
# ===============================
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Handle common PyTorch checkpoint layouts and strip prefixes."""
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            # Last resort: keep tensor-like items
            sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            if not sd:
                # flatten one level of nested dicts
                flat = {}
                for k, v in ckpt.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, torch.Tensor):
                                flat[f"{k}.{kk}"] = vv
                sd = flat or ckpt
    else:
        sd = ckpt

    clean = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("model."):
            nk = nk[6:]
        clean[nk] = v
    return clean


def _load_weights_flex(model: torch.nn.Module, path: str, device: torch.device) -> Tuple[List[str], List[str]]:
    ckpt = torch.load(path, map_location=device)
    sd = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return list(missing), list(unexpected)


# ===============================
# Tester
# ===============================
class QuartzNetTester:
    """
    If model_path ends with .nemo -> load NeMo ASR model and feed raw waveform.
    Else -> load local PyTorch QuartzNet and feed mel-spectrogram.
    """
    def __init__(self, model_path: str, processor: AudioProcessor, device: Optional[str] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.processor = processor
        self.model_path = model_path
        self.use_nemo = model_path.lower().endswith(".nemo")

        if self.use_nemo:
            if not _NEMO_OK:
                raise RuntimeError(
                    "NeMo is not installed. In your environment run: pip install nemo_toolkit[asr]"
                )
            self.nemo_model: ASRModel = ASRModel.restore_from(model_path, map_location=str(self.device))
            self.nemo_model.to(self.device)
            self.nemo_model.eval()
            print(f"[NeMo] Loaded ASR model from {model_path}")
        else:
            # Local PyTorch QuartzNet (your models/quartznet.py)
            self.model = QuartzNet()
            missing, unexpected = _load_weights_flex(self.model, model_path, self.device)
            if missing:
                print(f"[QuartzNet] Missing keys (first 20): {missing[:20]}{' …' if len(missing)>20 else ''}")
            if unexpected:
                print(f"[QuartzNet] Unexpected keys (first 20): {unexpected[:20]}{' …' if len(unexpected)>20 else ''}")
            self.model.to(self.device)
            self.model.eval()
            print(f"[Torch] Loaded QuartzNet weights from {model_path}")

    def transcribe(self, audio_path: str) -> str:
        """
        Returns a transcription string for a single audio file.
        """
        # Always load via your processor (handles resample → 16 kHz, mono)
        waveform = self.processor.load_audio(audio_path)  # torch.Tensor [T]
        if self.use_nemo:
            # NeMo accepts list of file paths; easiest & most accurate:
            preds = self.nemo_model.transcribe([audio_path])
            return preds[0] if preds else ""
        else:
            # Local mel + CTC greedy decode using your utilities
            mel = self.processor.audio_to_mel_spectrogram(waveform)  # [n_mels, T]
            mel = mel.unsqueeze(0).to(self.device)                   # [1, n_mels, T]
            with torch.no_grad():
                logits = self.model(mel)                             # [1, vocab, T]
            return self.processor.decode_predictions(logits[0])


# ===============================
# Metrics + saving
# ===============================
NORMALIZE = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
OUT_DIR = Path("outputs/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_metrics(reference: str, hypothesis: str) -> dict:
    raw = {"WER_raw": wer(reference, hypothesis), "CER_raw": cer(reference, hypothesis)}
    ref_n, hyp_n = NORMALIZE(reference), NORMALIZE(hypothesis)
    norm = {"WER_norm": wer(ref_n, hyp_n), "CER_norm": cer(ref_n, hyp_n)}
    return {**raw, **norm}

def save_result(audio_file: str, reference: str, hypothesis: str, model_name: str = "QuartzNet"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"wer_results_{ts}.json"
    metrics = compute_metrics(reference, hypothesis)
    data = {
        "audio_file": audio_file,
        "timestamp": ts,
        "transcription": hypothesis,
        "ground_truth": reference,
        "metrics": metrics,
        "model": model_name,
        "language": "english",
    }
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json, metrics

def evaluate_dataset(tester: QuartzNetTester, items: List[Dict[str, Any]]) -> dict:
    results = []
    for i, item in enumerate(items):
        ref, path = item["text"], item["audio_filepath"]
        hyp = tester.transcribe(path)
        out_json, m = save_result(path, ref, hyp)
        results.append({"idx": i, "audio": path, **m, "json": str(out_json)})
    # Write summary CSV
    if results:
        csv_path = OUT_DIR / f"results_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "audio_file", "WER_raw", "CER_raw", "WER_norm", "CER_norm", "json"])
            for r in results:
                w.writerow([r["idx"], r["audio"], r["WER_raw"], r["CER_raw"], r["WER_norm"], r["CER_norm"], r["json"]])
    return {"results": results}


# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        ckpt, audio, ref = sys.argv[1], sys.argv[2], " ".join(sys.argv[3:])
        proc = AudioProcessor()
        tester = QuartzNetTester(ckpt, proc)
        hyp = tester.transcribe(audio)
        out_json, m = save_result(audio, ref, hyp,
                                  model_name=("NeMo" if ckpt.lower().endswith(".nemo") else "QuartzNet"))
        print(f"Transcript: {hyp}")
        print(f"Metrics: {m}")
        print(f"Saved: {out_json}")
    else:
        print("Usage: python test.py <checkpoint.[pt|pth|nemo]> <audio.wav> <reference text>")
