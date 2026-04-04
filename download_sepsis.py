"""
Download PhysioNet Sepsis Challenge 2019 training data.
Run: python download_sepsis.py
"""
import os
import urllib.request
import re
import ssl

# Disable SSL verification (PhysioNet sometimes has cert issues)
ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"
OUTPUT_DIR = os.path.join("data", "mimic", "sepsis_challenge")

def download_set(set_name):
    """Download all .psv files from a training set."""
    set_dir = os.path.join(OUTPUT_DIR, set_name)
    os.makedirs(set_dir, exist_ok=True)

    url = BASE_URL + set_name + "/"
    print(f"\nFetching file list from {url}...")

    try:
        response = urllib.request.urlopen(url)
        html = response.read().decode()
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return

    # Extract .psv filenames from the HTML
    files = re.findall(r'href="(p\d+\.psv)"', html)
    total = len(files)
    print(f"Found {total} files in {set_name}")

    for i, fname in enumerate(files, 1):
        filepath = os.path.join(set_dir, fname)
        if os.path.exists(filepath):
            continue  # Skip already downloaded

        file_url = url + fname
        try:
            urllib.request.urlretrieve(file_url, filepath)
            if i % 500 == 0 or i == total:
                print(f"  [{i}/{total}] Downloaded {fname}")
        except Exception as e:
            print(f"  Error downloading {fname}: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("PhysioNet Sepsis Challenge 2019 Downloader")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    download_set("training_setA")
    download_set("training_setB")

    print("\nDone! Files saved to:", os.path.abspath(OUTPUT_DIR))
