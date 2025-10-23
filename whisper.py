#!/usr/bin/env python3
"""
transcribe_all_with_checkpoints.py

- Walks data/<accent> (auto-detects accents)
- Batched transcription with Whisper model
- Saves per-accent JSON files (progressive checkpoints)
- Uses IST timestamps
- Writes errors to error_log.txt
- Skips already-transcribed files so you can resume
- Stores JSON in the same structure you had:
    { "data/bangalore/audio_001.wav": {"transcription": "...", "timestamp": "YYYY-MM-DD HH:MM:SS"} }
"""

import os
import json
import time
import argparse
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
# Timezone (IST)
# -------------------------
IST = timezone(timedelta(hours=5, minutes=30))


def ist_now_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


# -------------------------
# Helpers: logging & JSON
# -------------------------
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
    """
    Atomic save. Sort keys so the JSON is deterministic/ordered.
    """
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as e:
        log_error(f"Failed atomic save for {path}: {e}")
        # cleanup tmp if exists
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise

# -------------------------
# Audio helper
# -------------------------
SAMPLE_RATE = 16000


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
# Path & accent helpers
# -------------------------
def accent_from_path(path, data_root):
    """
    Given a path like "data/bangalore/xyz.wav" return "bangalore".
    Works with both relative and absolute paths as long as data_root is present in parts.
    """
    parts = Path(path).parts
    try:
        if data_root in parts:
            idx = parts.index(data_root)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    # fallback: second part if path starts with something else
    return parts[1] if len(parts) >= 2 else "unknown"


def collect_audio_paths(data_root):
    """
    Auto-detect top-level subfolders under data_root and collect sorted .wav paths.
    Returns sorted list of normalized relative paths (relative to cwd) for deterministic ordering.
    """
    base = Path(data_root)
    if not base.exists() or not base.is_dir():
        print(f"Data root not found or not a directory: {data_root}")
        return []

    paths = []
    # top-level subfolders only (accents)
    for accent_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        # walk inside accent_dir to collect .wav files
        for root, _, files in os.walk(accent_dir):
            for fname in sorted(files):
                if fname.lower().endswith(".wav"):
                    # store as normalized relative path (like data/bangalore/file.wav)
                    rel = os.path.normpath(os.path.join(root, fname))
                    paths.append(rel)
    # deterministic global sort (though we already sorted by accent_dir and filenames)
    paths = sorted(paths)
    return paths


# -------------------------
# Main transcription logic
# -------------------------
def batch_transcribe_and_checkpoint(
    audio_paths,
    data_root,
    processor,
    model,
    device,
    output_template="transcriptions_{}.json",
    batch_size=8,
    checkpoint_interval=1,
    verbose=True,
):
    # load existing per-accent JSONs for accents discovered in data_root
    accents = sorted({accent_from_path(p, data_root) for p in audio_paths})
    per_accent_data = {accent: safe_load_json(output_template.format(accent)) for accent in accents}

    total_files = len(audio_paths)
    if verbose:
        print(f"Detected accents: {accents}")
        print(f"Found {total_files} audio files in '{data_root}' (across accents).")

    # helper to count completed entries across all per_accent_data
    def count_done():
        return sum(1 for accent in per_accent_data for k in per_accent_data.get(accent, {}) if per_accent_data[accent].get(k, {}).get("transcription"))

    done_count = count_done()

    for i in tqdm(range(0, total_files, batch_size), desc="Batches"):
        batch = audio_paths[i : i + batch_size]

        to_process = []
        for p in batch:
            accent = accent_from_path(p, data_root)
            # ensure per_accent_data has the accent (might be new)
            if accent not in per_accent_data:
                per_accent_data[accent] = {}
            perjson = per_accent_data.get(accent, {})
            rel_key = os.path.normpath(p)
            existing = perjson.get(rel_key)
            if existing and existing.get("transcription"):
                if verbose:
                    print(f"Skipping already transcribed: {rel_key}")
                continue
            to_process.append((p, rel_key, accent))

        if not to_process:
            # still print summary occasionally
            if verbose:
                print(f"No files to process in batch starting at index {i}. Done: {done_count}/{total_files}")
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
            if verbose:
                print(f"No valid audio in batch starting at {i}.")
            continue

        # Prepare inputs
        try:
            inputs = processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            input_features = inputs.input_features.to(device)
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
            done_count += 1

        # Save checkpoints for touched accents
        touched = sorted({a for (_, _, a) in valid_items})
        for accent in touched:
            outjson = output_template.format(accent)
            try:
                save_json_atomic(outjson, per_accent_data.get(accent, {}))
                if verbose:
                    print(f"Checkpoint saved: {outjson} (entries: {len(per_accent_data[accent])})")
            except Exception as e:
                log_error(f"Failed to save checkpoint {outjson}: {e}")

        # Summary after each batch
        if verbose:
            print(f"✅ {done_count}/{total_files} files done (after batch ending at index {i + len(batch) - 1})")

        # small sleep to be gentle on I/O
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="Batched Whisper transcription with per-accent checkpoints (resumable).")
    parser.add_argument("--model", type=str, default="ARTPARK-IISc/whisper-medium-vaani-kannada", help="HuggingFace model name")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory containing accent subfolders")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run model on (cuda or cpu)")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="(unused) kept for compatibility; script saves each batch")
    parser.add_argument("--verbose", action="store_true", help="Verbose prints")
    args = parser.parse_args()

    MODEL_NAME = args.model
    DATA_ROOT = args.data_dir
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    VERBOSE = args.verbose

    # Prepare model & processor
    print(f"Loading model '{MODEL_NAME}' to {DEVICE} ...")
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    except Exception as e:
        log_error(f"Failed to load feature extractor from '{MODEL_NAME}': {e}")
        raise

    try:
        # keep tokenizer as the openai base but set language/task as before
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Kannada", task="transcribe")
    except Exception as e:
        log_error(f"Failed to load tokenizer: {e}")
        raise

    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    try:
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
    except Exception as e:
        log_error(f"Failed to load model '{MODEL_NAME}': {e}")
        raise

    print("Model loaded.")

    # Gather audio
    audio_paths = collect_audio_paths(DATA_ROOT)
    if not audio_paths:
        print("No audio files found. Make sure the data folders exist and contain .wav files.")
        return

    # Run transcription & checkpointing
    batch_transcribe_and_checkpoint(
        audio_paths,
        data_root=DATA_ROOT,
        processor=processor,
        model=model,
        device=DEVICE,
        output_template="transcriptions_{}.json",
        batch_size=BATCH_SIZE,
        checkpoint_interval=args.checkpoint_interval,
        verbose=VERBOSE,
    )

    print("All done. Per-accent JSONs written (check current directory).")
    print(f"Any errors were appended to '{ERROR_LOG}'.")


if __name__ == "__main__":
    main()