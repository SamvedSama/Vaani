import os
import json
import csv
import requests
from tqdm import tqdm

# ========== USER CONFIG ==========
JSON_FILE = "./data/vaani.json"
DATA_DIR = "data"
CSV_FILE = "accent_metadata.csv"
TARGET_DISTRICTS = ["bangalore", "bidar", "dharwad", "mysore"]  # lowercase match
LIMIT_PER_DISTRICT = 50
# =================================

def download_file(url, dest_path):
    """Download a file if not already present."""
    if os.path.exists(dest_path):
        return
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    except Exception as e:
        print(f" Error downloading {url}: {e}")

def main():
    # Load JSON
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Track counts per district
    district_counts = {d: 0 for d in TARGET_DISTRICTS}
    csv_rows = []

    for entry in tqdm(data, desc="Processing dataset"):
        metadata = entry.get("metadata", {})
        district = metadata.get("district", "").strip().lower()

        if district not in TARGET_DISTRICTS:
            continue

        # Stop if we already have enough
        if district_counts[district] >= LIMIT_PER_DISTRICT:
            continue

        file_name = entry.get("file_name")
        file_url = entry.get("file_url")

        if not file_name or not file_url:
            continue

        # District folder
        district_folder = os.path.join(DATA_DIR, district)
        os.makedirs(district_folder, exist_ok=True)

        # Local path
        local_path = os.path.join(district_folder, file_name)

        # Download
        download_file(file_url, local_path)

        # Add entry to CSV (reference left blank)
        csv_rows.append([district, local_path, ""])
        district_counts[district] += 1

        # If all districts have reached the limit, break early
        if all(count >= LIMIT_PER_DISTRICT for count in district_counts.values()):
            break

    # Write CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["accent", "audio_path", "reference"])
        writer.writerows(csv_rows)

    print("\nFinished!")
    print(" Per-district counts:", district_counts)
    print(f" Audio saved under '{DATA_DIR}/<district>/'")
    print(f" Metadata file created: {CSV_FILE}")

if __name__ == "__main__":
    main()
