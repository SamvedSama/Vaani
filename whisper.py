"""
transcribe_all_with_checkpoints.py

- Walks data/{bangalore, mysore, bidar, dharwad}
- Batched transcription with Whisper model
- Saves per-accent JSON files (progressive checkpoints)
- Uses IST timestamps
- Writes errors to error_log.txt
- Skips already-transcribed files so you can resume
"""

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
    WhisperFeatureExtractor,
)
from tqdm import tqdm

# Minimize vision backend usage to avoid torchvision import on Whisper-only pipelines
os.environ.setdefault("HF_USE_VISION_BACKEND", "0")

# Optional: if torchvision is installed, importing it early can register fake ops and avoid nms error
try:
    import torchvision  # noqa: F401
except Exception:
    pass

# -------------------------
# Configuration
# -------------------------
MODEL_NAME = "ARTPARK-IISc/whisper-medium-vaani-kannada"  # your model
DATA_ROOT = "data"
TARGET_FOLDERS = ["bangalore", "mysore", "bidar", "dharwad"]  # lowercase folder names
BATCH_SIZE = 8
OUTPUT_JSON_TEMPLATE = "transcriptions_{}.json"  # e.g., transcriptions_bangalore.json
ERROR_LOG = "error_log.txt"
CHECKPOINT_INTERVAL = 1  # currently save every batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000

# IST timezone for timestamps
IST = timezone(timedelta(hours=5, minutes=30))

# -------------------------
# Utilities
# -------------------------
def ist_now_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def log_error(msg):
    timestamp = ist_now_str()
    entry = f"[{timestamp}] {msg}\n"
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(entry)
    print("ERROR:", msg)

def safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            backup = f"{path}.broken_{int(time.time())}"
            try:
                os.rename(path, backup)
            except Exception:
                pass
            log_error(f"Backed up corrupted JSON {path} -> {backup}")
            return {}
    return {}

def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# -------------------------
# Model & Processor Load
# -------------------------
print(f"Loading model '{MODEL_NAME}' to {DEVICE} ...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Kannada", task="transcribe")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded.")

# -------------------------
# Audio helpers
# -------------------------
def load_and_resample(path, target_sr=SAMPLE_RATE):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        try:
            tensor = torch.tensor(data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            resampled = resampler(tensor).squeeze(0).numpy()
            return resampled
        except Exception as e:
            try:
                import librosa
                resampled = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=target_sr)
                return resampled
            except Exception as e2:
                raise RuntimeError(f"Resampling failed for {path}: {e} / {e2}")
    return data

# -------------------------
# Gather files
# -------------------------
def collect_audio_paths():
    paths = []
    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(DATA_ROOT, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: folder not found: {folder_path} (skipping)")
            continue
        for root, _, files in os.walk(folder_path):
            for f in sorted(files):
                if f.lower().endswith(".wav"):
                    paths.append(os.path.join(root, f))
    return paths

# -------------------------
# Per-accent JSON handling
# -------------------------
def accent_from_path(path):
    parts = Path(path).parts
    # expect data/<accent>/...
    try:
        # find DATA_ROOT segment index
        if DATA_ROOT in parts:
            idx = parts.index(DATA_ROOT)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    # fallback
    return parts[1] if len(parts) >= 2 else "unknown"

def get_output_json_for_accent(accent):
    return OUTPUT_JSON_TEMPLATE.format(accent)

# -------------------------
# Transcription loop (batched)
# -------------------------
def batch_transcribe_and_checkpoint(audio_paths, batch_size=BATCH_SIZE):
    per_accent_data = {accent: safe_load_json(get_output_json_for_accent(accent)) for accent in TARGET_FOLDERS}

    total = len(audio_paths)
    print(f"Found {total} audio files across target folders.")

    for i in tqdm(range(0, total, batch_size), desc="Batches"):
        batch = audio_paths[i : i + batch_size]

        # Skip already transcribed
        to_process = []
        for p in batch:
            accent = accent_from_path(p)
            perjson = per_accent_data.get(accent, {})
            rel_key = os.path.normpath(p)
            if rel_key in perjson and perjson[rel_key].get("transcription"):
                continue
            to_process.append((p, rel_key, accent))

        if not to_process:
            continue

        # Load audio
        audios = []
        valid_items = []
        for p, rel_key, accent in to_process:
            try:
                audio_np = load_and_resample(p, SAMPLE_RATE)
                audios.append(audio_np)
                valid_items.append((p, rel_key, accent))
            except Exception as e:
                log_error(f"Failed to load/resample '{p}': {e}")

        if not audios:
            continue

        # Prepare inputs
        try:
            inputs = processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            input_features = inputs.input_features.to(DEVICE)
        except Exception as e:
            log_error(f"Processor error for batch starting at {i}: {e}")
            continue

        # Generate
        try:
            with torch.no_grad():
                pred_ids = model.generate(input_features, max_length=225)
            trans_list = processor.batch_decode(pred_ids, skip_special_tokens=True)
        except Exception as e:
            log_error(f"Model generation error for batch starting at {i}: {e}")
            continue

        ts = ist_now_str()
        for (p, rel_key, accent), transcription in zip(valid_items, trans_list):
            entry = {"transcription": transcription.strip(), "timestamp": ts}
            if accent not in per_accent_data:
                per_accent_data[accent] = {}
            per_accent_data[accent][rel_key] = entry

        # Progressive checkpoint per accent touched
        touched = {a for (_, _, a) in valid_items}
        for accent in touched:
            outjson = get_output_json_for_accent(accent)
            try:
                save_json_atomic(outjson, per_accent_data.get(accent, {}))
            except Exception as e:
                log_error(f"Failed to save checkpoint {outjson}: {e}")

        time.sleep(0.01)

# -------------------------
# Main
# -------------------------
def main():
    all_audio = collect_audio_paths()
    if not all_audio:
        print("No audio files found. Make sure the data folders exist and contain .wav files.")
        return

    all_audio = sorted(all_audio)
    batch_transcribe_and_checkpoint(all_audio, batch_size=BATCH_SIZE)

    print("All done. Per-accent JSONs written (check current directory).")
    print(f"Any errors were appended to '{ERROR_LOG}'.")

if __name__ == "__main__":
    main()