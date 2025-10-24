import os
import json
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
JSON_FILE = "./data/vaani.json"
DATA_DIR = "data"
PROGRESS_FILE = "progress.json"
LIMIT_PER_DISTRICT = 1000
ALLOWED_DISTRICTS = {"bangalore", "mysore", "bidar", "dharwad"}


# -------------------------
# Helpers
# -------------------------
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
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}  # district -> start_index


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# -------------------------
# Main
# -------------------------
def main():
    # Load dataset
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by district
    district_data = {}
    for entry in data:
        district = entry.get("metadata", {}).get("district", "unknown").strip().lower()
        if district in ALLOWED_DISTRICTS:
            district_data.setdefault(district, []).append(entry)

    progress = load_progress()

    for district, entries in tqdm(district_data.items(), desc="Processing districts"):
        start_index = progress.get(district, 0)
        end_index = start_index + LIMIT_PER_DISTRICT

        # Slice next batch
        batch = entries[start_index:end_index]
        if not batch:
            print(f"No more entries left for {district}.")
            continue

        # Determine folder for this batch
        # Determine folder for this batch inside the district
        district_base = os.path.join(DATA_DIR, district)
        os.makedirs(district_base, exist_ok=True)  # make sure district folder exists

        batch_number = start_index // LIMIT_PER_DISTRICT + 1
        district_folder = os.path.join(district_base, str(batch_number))  # batch folder as subfolder


        # Create new folder
        if os.path.exists(district_folder):
            # Remove duplicates only, keep folder for consistency
            existing_files = {f for f in os.listdir(district_folder)}
        else:
            os.makedirs(district_folder, exist_ok=True)
            existing_files = set()

        # Download batch
        for entry in batch:
            file_name = entry.get("file_name")
            file_url = entry.get("file_url")
            if not file_name or not file_url:
                continue

            if file_name in existing_files:
                continue  # skip duplicates

            local_path = os.path.join(district_folder, file_name)
            download_file(file_url, local_path)
            existing_files.add(file_name)

        # Update progress
        progress[district] = end_index

    # Save progress
    save_progress(progress)
    print("✅ Download complete. Progress saved in progress.json")


if __name__ == "__main__":
    main()
