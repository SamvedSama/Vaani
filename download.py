import os
import json
import csv
import requests
import shutil
from tqdm import tqdm

# ========== USER CONFIG ==========
JSON_FILE = "./data/vaani.json"
DATA_DIR = "data"
CSV_FILE = "accent_metadata.csv"
PROGRESS_FILE = "progress.json"
LIMIT_PER_DISTRICT = 1000  # how many to download per run per district
ALLOWED_DISTRICTS = {"bangalore", "mysore", "bidar", "dharwad"}
# =================================


def download_file(url, dest_path):
    """Download a file if not already present."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    except Exception as e:
        print(f" Error downloading {url}: {e}")


def load_progress():
    """Load or initialize district index tracking."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}  # district -> start_index


def save_progress(progress):
    """Save updated district index tracking."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def main():
    # Load dataset JSON
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pre-group dataset by allowed districts only
    district_data = {}
    for entry in data:
        district = entry.get("metadata", {}).get("district", "unknown").strip().lower()
        if district in ALLOWED_DISTRICTS:
            district_data.setdefault(district, []).append(entry)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Load or create progress tracker
    progress = load_progress()
    csv_rows = []

    # Process each allowed district
    for district, entries in tqdm(district_data.items(), desc="Processing districts"):

        start_index = progress.get(district, 0)
        end_index = start_index + LIMIT_PER_DISTRICT
        subset = entries[start_index:end_index]

        # Prepare district folder (delete old before new batch)
        district_folder = os.path.join(DATA_DIR, district)
        if os.path.exists(district_folder):
            shutil.rmtree(district_folder)
        os.makedirs(district_folder, exist_ok=True)

        # Download this batch
        for entry in subset:
            file_name = entry.get("file_name")
            file_url = entry.get("file_url")

            if not file_name or not file_url:
                continue

            local_path = os.path.join(district_folder, file_name)
            download_file(file_url, local_path)

            csv_rows.append([district, local_path, ""])

        # Update progress index
        if end_index >= len(entries):
            progress[district] = 0  # reset if finished, or change logic if you prefer to stop
        else:
            progress[district] = end_index

    # Save progress to disk
    save_progress(progress)

    # Write updated CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["accent", "audio_path", "reference"])
        writer.writerows(csv_rows)

    print("\n✅ Finished!")
    print(f"📂 Audio saved under '{DATA_DIR}/<district>/'")
    print(f"📄 Metadata file created: {CSV_FILE}")
    print(f"📌 Progress saved in: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
