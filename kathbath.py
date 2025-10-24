import os
import json
import csv
from datetime import datetime, timezone, timedelta

import torch
import soundfile as sf
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from tqdm import tqdm
from jiwer import wer, cer

# -------------------------
# PHONEME SUPPORT
# -------------------------
try:
    from phonemizer import phonemize
    PHONEME_AVAILABLE = True
except ImportError:
    print("⚠ Phonemizer not installed. PER will be skipped.")
    PHONEME_AVAILABLE = False

import subprocess

# -------------------------
# CHECK eSpeak NG
# -------------------------
ESPEAK_PATH = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
if os.path.exists(ESPEAK_PATH):
    try:
        result = subprocess.run([ESPEAK_PATH, "--version"], capture_output=True, text=True)
        print(f"✅ eSpeak NG detected: {result.stdout.strip()}")
        ESPEAK_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] eSpeak NG exists but failed to run: {e}")
        ESPEAK_AVAILABLE = False
else:
    print(f"⚠ eSpeak NG not found at {ESPEAK_PATH}. PER will be skipped.")
    ESPEAK_AVAILABLE = False

# -------------------------
# CONFIG
# -------------------------
DEFAULT_CONFIG = {
    "model_name": "ARTPARK-IISc/whisper-medium-vaani-kannada",
    "sample_rate": 16000,
    "phoneme_backend": "espeak",
    "phoneme_language": "kn",
    "accent_label": "kathbath",
    "batch_size": 2
}

PHONEME_LANG = DEFAULT_CONFIG["phoneme_language"]
ACCENT_LABEL = DEFAULT_CONFIG["accent_label"]
BATCH_SIZE = DEFAULT_CONFIG["batch_size"]

# Enable Windows espeak-ng for phonemizer if available
if PHONEME_AVAILABLE and ESPEAK_AVAILABLE:
    os.environ["PHONEMIZER_ESPEAK_PATH"] = ESPEAK_PATH

PARAMS_PATH = "params.json"
if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        DEFAULT_CONFIG.update(json.load(f))

MODEL_NAME = DEFAULT_CONFIG["model_name"]
SAMPLE_RATE = DEFAULT_CONFIG["sample_rate"]
PHONEME_LANG = DEFAULT_CONFIG["phoneme_language"]
ACCENT_LABEL = DEFAULT_CONFIG["accent_label"]
BATCH_SIZE = DEFAULT_CONFIG["batch_size"]

DATASET_ROOT = "./data/Kathbath-Kannada-Test-Known"
DATA_JSON_PATH = os.path.join(DATASET_ROOT, "data.json")

OUTPUT_JSON = "transcriptions_kathbath_test.json"
OUTPUT_CSV = "transcriptions_kathbath_test.csv"
METRICS_JSON = "metrics_report.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IST = timezone(timedelta(hours=5, minutes=30))

# -------------------------
# HELPERS
# -------------------------
def ist_now_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def write_csv(data, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audioFilename", "reference_text", "predicted_text", "timestamp"])
        for k, v in data.items():
            writer.writerow([k, v["reference_text"], v["predicted_text"], v["timestamp"]])

def load_resample(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        data = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(torch.tensor(data)).numpy()
    return data

# -------------------------
# LOAD DATA
# -------------------------
with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    dataset_entries = json.load(f)

reference_map = {entry["audioFilename"]: entry["text"] for entry in dataset_entries}
audio_files = list(reference_map.keys())

# -------------------------
# TRANSCRIPTION (BATCHED)
# -------------------------
# -------------------------
# TRANSCRIPTION (BATCHED)
# -------------------------
results = {}

# Check if transcriptions exist and are complete
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    missing_files = [f for f in audio_files if f not in results]
    if not missing_files:
        print(f"ℹ All transcriptions found in {OUTPUT_JSON}. Skipping transcription.")
    else:
        print(f"⚠ Missing {len(missing_files)} transcriptions. Transcribing missing files...")
        audio_files = missing_files  # Only transcribe missing files
        results = {k: v for k, v in results.items() if k in results}  # Keep existing
        run_transcription = True
else:
    run_transcription = True

if 'run_transcription' in locals() and run_transcription:
    print(f"📦 Starting batched transcription ({len(audio_files)} files)...")
    # Load model
    print(f"🚀 Loading model '{MODEL_NAME}' on {DEVICE} ...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Kannada", task="transcribe")
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("✅ Model loaded.")

    for i in tqdm(range(0, len(audio_files), BATCH_SIZE), desc="Batches"):
        batch_files = audio_files[i:i + BATCH_SIZE]
        audio_inputs = []

        for fname in batch_files:
            audio_path = os.path.join(DATASET_ROOT, fname)
            if os.path.exists(audio_path):
                audio_inputs.append(load_resample(audio_path))
            else:
                print(f"[WARN] Missing: {fname}")
                audio_inputs.append(None)

        inputs = processor([a for a in audio_inputs if a is not None],
                           sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            pred_ids = model.generate(inputs.input_features, max_length=225)

        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)

        for fname, pred in zip(batch_files, predictions):
            results[fname] = {
                "reference_text": reference_map[fname],
                "predicted_text": pred.strip(),
                "timestamp": ist_now_str()
            }

    save_json_atomic(OUTPUT_JSON, results)
    write_csv(results, OUTPUT_CSV)


# -------------------------
# PER-AUDIO METRICS
# -------------------------
print("✅ Computing per-audio metrics...")

per_audio_metrics = {}

for fname, res in results.items():
    ref = res["reference_text"]
    pred = res["predicted_text"]

    # WER & CER
    wer_val = wer([ref], [pred])
    cer_val = cer([ref], [pred])
    acc_val = 1.0 if ref == pred else 0.0

    # PER (optional)
    if PHONEME_AVAILABLE and ESPEAK_AVAILABLE:
        try:
            refs_ph = phonemize(ref, language=PHONEME_LANG, backend="espeak", strip=True, preserve_punctuation=True)
            preds_ph = phonemize(pred, language=PHONEME_LANG, backend="espeak", strip=True, preserve_punctuation=True)
            per_val = cer([refs_ph], [preds_ph])
        except Exception as e:
            print(f"[WARN] PER computation failed for {fname}: {e}")
            per_val = None
    else:
        per_val = None

    per_audio_metrics[fname] = {
        "wer": wer_val,
        "cer": cer_val,
        "accuracy": acc_val,
        "per": per_val
    }

# -------------------------
# Overall metrics
# -------------------------
overall_wer = sum(m["wer"] for m in per_audio_metrics.values()) / len(per_audio_metrics)
overall_cer = sum(m["cer"] for m in per_audio_metrics.values()) / len(per_audio_metrics)
overall_accuracy = sum(m["accuracy"] for m in per_audio_metrics.values()) / len(per_audio_metrics)
overall_per = None
if PHONEME_AVAILABLE and ESPEAK_AVAILABLE:
    per_values = [m["per"] for m in per_audio_metrics.values() if m["per"] is not None]
    overall_per = sum(per_values)/len(per_values) if per_values else None

metrics = {
    "accent": ACCENT_LABEL,
    "overall": {
        "wer": overall_wer,
        "cer": overall_cer,
        "accuracy": overall_accuracy,
        "per": overall_per
    },
    "per_audio": per_audio_metrics
}

# Save metrics
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("📊 Metrics saved:", METRICS_JSON)
