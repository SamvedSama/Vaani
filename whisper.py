#!/usr/bin/env python3
"""
transcribe_all_with_checkpoints.py (global batch for all districts)

- Reads audio files from structured folders created by previous download script
- Uses global variable BATCH_NUM to pick batch folder for all districts
- Example: BATCH_NUM = 5 → data/bangalore/5, data/mysore/5, etc.
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

os.environ.setdefault("HF_USE_VISION_BACKEND", "0")

try:
    import torchvision  # noqa: F401
except Exception:
    pass

# -------------------------
# GLOBAL CONFIG
# -------------------------
BATCH_NUM = 1  # <--- change this number to pick batch for all districts 
DISTRICTS = ["bangalore", "mysore", "bidar", "dharwad"]
DATA_ROOT = "data"
SAMPLE_RATE = 16000
MODEL_NAME = "ARTPARK-IISc/whisper-medium-vaani-kannada"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
VERBOSE = True

IST = timezone(timedelta(hours=5, minutes=30))


# -------------------------
# Utils
# -------------------------
def ist_now_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


ERROR_LOG = "error_log.txt"


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
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as e:
        log_error(f"Failed atomic save for {path}: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


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


def accent_from_path(path, data_root):
    parts = Path(path).parts
    try:
        if data_root in parts:
            idx = parts.index(data_root)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    return parts[1] if len(parts) >= 2 else "unknown"


def collect_audio_paths_for_batch(data_root, batch_num):
    """
    Collect audio files from the specified batch number folder for all districts.
    """
    paths = []
    for district in DISTRICTS:
        batch_folder = os.path.join(data_root, district, str(batch_num))
        if not os.path.exists(batch_folder):
            log_error(f"Batch folder not found: {batch_folder}")
            continue
        for root, _, files in os.walk(batch_folder):
            for fname in sorted(files):
                if fname.lower().endswith(".wav"):
                    paths.append(os.path.normpath(os.path.join(root, fname)))
    return sorted(paths)


# -------------------------
# Transcription
# -------------------------
def batch_transcribe_and_checkpoint(audio_paths, processor, model, device):
    accents = sorted({accent_from_path(p, DATA_ROOT) for p in audio_paths})
    per_accent_data = {accent: safe_load_json(f"transcriptions_{accent}.json") for accent in accents}
    total_files = len(audio_paths)
    if VERBOSE:
        print(f"Detected accents: {accents}")
        print(f"Found {total_files} audio files for batch {BATCH_NUM}.")

    done_count = sum(
        1
        for accent in per_accent_data
        for k in per_accent_data.get(accent, {})
        if per_accent_data[accent].get(k, {}).get("transcription")
    )

    for i in tqdm(range(0, total_files, BATCH_SIZE), desc="Batches"):
        batch = audio_paths[i : i + BATCH_SIZE]
        to_process = []
        for p in batch:
            accent = accent_from_path(p, DATA_ROOT)
            if accent not in per_accent_data:
                per_accent_data[accent] = {}
            perjson = per_accent_data.get(accent, {})
            rel_key = os.path.normpath(p)
            existing = perjson.get(rel_key)
            if existing and existing.get("transcription"):
                if VERBOSE:
                    print(f"Skipping already transcribed: {rel_key}")
                continue
            to_process.append((p, rel_key, accent))

        if not to_process:
            continue

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

        try:
            inputs = processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            input_features = inputs.input_features.to(device)
        except Exception as e:
            log_error(f"Processor error for batch starting at {i}: {e}")
            continue

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
            per_accent_data.setdefault(accent, {})[rel_key] = entry
            done_count += 1

        touched = sorted({a for (_, _, a) in valid_items})
        for accent in touched:
            outjson = f"transcriptions_{accent}.json"
            try:
                save_json_atomic(outjson, per_accent_data.get(accent, {}))
                if VERBOSE:
                    print(f"Checkpoint saved: {outjson} (entries: {len(per_accent_data[accent])})")
            except Exception as e:
                log_error(f"Failed to save checkpoint {outjson}: {e}")

        if VERBOSE:
            print(f"✅ {done_count}/{total_files} files done (after batch ending at index {i + len(batch) - 1})")

        time.sleep(0.01)


def main():
    print(f"Loading model '{MODEL_NAME}' to {DEVICE} ...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Kannada", task="transcribe")
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("Model loaded.")

    audio_paths = collect_audio_paths_for_batch(DATA_ROOT, BATCH_NUM)
    if not audio_paths:
        print(f"No audio files found for batch {BATCH_NUM}. Check that the folders exist.")
        return

    batch_transcribe_and_checkpoint(audio_paths, processor, model, DEVICE)
    print(f"All done for batch {BATCH_NUM}. Per-accent JSONs written.")
    print(f"Any errors were appended to '{ERROR_LOG}'.")


if __name__ == "__main__":
    main()