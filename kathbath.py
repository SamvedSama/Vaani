import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

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

try:
    from phonemizer import phonemize
    PHONEME_AVAILABLE = True
except ImportError:
    print(" Phonemizer not installed. PER will be skipped.")
    PHONEME_AVAILABLE = False

# -------------------------
# LOAD PARAMS (OPTIONAL)
# -------------------------
DEFAULT_CONFIG = {
    "model_name": "ARTPARK-IISc/whisper-medium-vaani-kannada",
    "sample_rate": 16000,
    "phoneme_backend": "espeak",
    "phoneme_language": "kn",
    "accent_label": "kathbath"
}

PARAMS_PATH = "params.json"
if os.path.exists(PARAMS_PATH):
    print(f" Loading optional config from {PARAMS_PATH}")
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        user_params = json.load(f)
        DEFAULT_CONFIG.update(user_params)
else:
    print("ℹ No params.json found. Using defaults.")

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_NAME = DEFAULT_CONFIG["model_name"]
SAMPLE_RATE = DEFAULT_CONFIG["sample_rate"]
PHONEME_BACKEND = DEFAULT_CONFIG["phoneme_backend"]
PHONEME_LANG = DEFAULT_CONFIG["phoneme_language"]
ACCENT_LABEL = DEFAULT_CONFIG["accent_label"]

DATASET_ROOT = "data/Kathbath-Kannada-Test-Known"
DATA_JSON_PATH = os.path.join(DATASET_ROOT, "data.json")
OUTPUT_JSON = "transcriptions_kathbath_test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IST = timezone(timedelta(hours=5, minutes=30))

# -------------------------
# TIME HELPERS
# -------------------------
def ist_now_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# -------------------------
# JSON HELPERS
# -------------------------
def safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            backup = f"{path}.broken_{int(time.time())}"
            os.rename(path, backup)
            print(f"[WARN] Corrupted JSON backed up to {backup}")
            return {}
    return {}

def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# -------------------------
# MODEL & PROCESSOR LOAD
# -------------------------
print(f"Loading model '{MODEL_NAME}' on {DEVICE} ...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Kannada", task="transcribe")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(" Model ready.")

# -------------------------
# AUDIO LOADING
# -------------------------
def load_and_resample(path, target_sr=SAMPLE_RATE):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        tensor = torch.tensor(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        data = resampler(tensor).squeeze(0).numpy()
    return data

# -------------------------
# LOAD GROUND TRUTH
# -------------------------
with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    dataset_entries = json.load(f)

reference_map = {entry["audioFilename"]: entry["text"] for entry in dataset_entries}
audio_files = [f for f in reference_map.keys() if f.lower().endswith(".wav")]
print(f" Found {len(audio_files)} audio files in data.json.")

# -------------------------
# LOAD EXISTING PROGRESS
# -------------------------
results = safe_load_json(OUTPUT_JSON)

# -------------------------
# TRANSCRIPTION LOOP
# -------------------------
for audio_filename in tqdm(audio_files, desc="Transcribing"):
    if audio_filename in results and "predicted_text" in results[audio_filename]:
        continue

    audio_path = os.path.join(DATASET_ROOT, audio_filename)
    if not os.path.exists(audio_path):
        print(f"[WARN] Missing audio: {audio_path}, skipping")
        continue

    try:
        audio_np = load_and_resample(audio_path)
        inputs = processor([audio_np], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(DEVICE)

        with torch.no_grad():
            pred_ids = model.generate(input_features, max_length=225)

        predicted = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        results[audio_filename] = {
            "reference_text": reference_map[audio_filename],
            "predicted_text": predicted,
            "timestamp": ist_now_str()
        }

        save_json_atomic(OUTPUT_JSON, results)

    except Exception as e:
        print(f"[ERROR] {audio_filename}: {e}")
        continue

print(" Transcription done! Computing metrics...")

# -------------------------
# METRIC CALCULATION
# -------------------------
refs = [entry["reference_text"] for entry in results.values()]
preds = [entry["predicted_text"] for entry in results.values()]

global_wer = wer(refs, preds)
global_cer = cer(refs, preds)

if PHONEME_AVAILABLE:
    ref_phonemes = phonemize(refs, language=PHONEME_LANG, backend=PHONEME_BACKEND, strip=True)
    pred_phonemes = phonemize(preds, language=PHONEME_LANG, backend=PHONEME_BACKEND, strip=True)
    per = cer(ref_phonemes, pred_phonemes)
else:
    per = None

accuracy = sum(1 for r, p in zip(refs, preds) if r == p) / len(refs)

metrics_report = {
    "accent": ACCENT_LABEL,
    "global_wer": global_wer,
    "global_cer": global_cer,
    "phoneme_error_rate": per,
    "accent_accuracy": accuracy
}

with open("metrics_report.json", "w", encoding="utf-8") as f:
    json.dump(metrics_report, f, indent=2, ensure_ascii=False)

print(" Metrics saved to metrics_report.json")
print(metrics_report)