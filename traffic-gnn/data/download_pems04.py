"""
Download PEMS04 dataset.

Auto-downloads the .npz and adjacency CSV from a public mirror.
If the mirror is down, prints manual instructions.

Usage
-----
    python data/download_pems04.py
"""

import os
import sys
import zipfile
import requests
from pathlib import Path

# ── URLs (ASTGCN public mirror on GitHub) ────────────────────────────────────
BASE_URL = "https://github.com/guoshnBJTU/ASTGCN-r-pytorch/raw/main/data/PEMS04"
FILES = {
    "pems04.npz": f"{BASE_URL}/pems04.npz",
    "pems04.csv": f"{BASE_URL}/distance.csv",
}

DATA_DIR = Path(__file__).resolve().parent


def download_file(url: str, dest: Path, desc: str = ""):
    """Download with progress bar."""
    print(f"  Downloading {desc or dest.name} ...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  [{pct:5.1f}%] {downloaded:,} / {total:,} bytes", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def try_alternative_sources():
    """Try multiple known mirrors for PEMS04."""
    alt_urls = [
        # Mirror 1: STGCN repo
        {
            "pems04.npz": "https://github.com/hazdzz/STGCN/raw/main/data/PEMS04/pems04.npz",
            "pems04.csv": "https://github.com/hazdzz/STGCN/raw/main/data/PEMS04/distance.csv",
        },
        # Mirror 2: AGCRN repo
        {
            "pems04.npz": "https://github.com/LeiBAI/AGCRN/raw/main/data/PEMS04/pems04.npz",
            "pems04.csv": "https://github.com/LeiBAI/AGCRN/raw/main/data/PEMS04/distance.csv",
        },
    ]
    for i, urls in enumerate(alt_urls, 1):
        print(f"\n  Trying mirror {i} ...")
        success = True
        for fname, url in urls.items():
            dest = DATA_DIR / fname
            if not download_file(url, dest, fname):
                success = False
                break
        if success:
            return True
    return False


def print_manual_instructions():
    """Print instructions for manual download."""
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║              MANUAL DOWNLOAD INSTRUCTIONS                    ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  Auto-download failed. Please download manually:             ║
  ║                                                              ║
  ║  1. Go to one of these repositories:                         ║
  ║     • github.com/guoshnBJTU/ASTGCN-r-pytorch                ║
  ║     • github.com/hazdzz/STGCN                               ║
  ║                                                              ║
  ║  2. Download `pems04.npz` and `distance.csv`                 ║
  ║     from the data/PEMS04/ folder                             ║
  ║                                                              ║
  ║  3. Place them in the data/ directory:                       ║
  ║       data/pems04.npz                                        ║
  ║       data/pems04.csv                                        ║
  ║                                                              ║
  ║  Expected shape: (16992, 307, 3)                             ║
  ║  Features: flow, occupancy, speed                            ║
  ╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    print("\n" + "=" * 60)
    print("  PEMS04 Dataset Downloader")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if already exists
    npz_path = DATA_DIR / "pems04.npz"
    csv_path = DATA_DIR / "pems04.csv"

    if npz_path.exists() and csv_path.exists():
        print(f"\n  ✓ Data already exists at {DATA_DIR}")
        # Verify
        import numpy as np
        d = np.load(npz_path)
        print(f"  ✓ pems04.npz shape: {d['data'].shape}")
        return

    # Try primary URL
    print("\n  Attempting download from primary mirror ...")
    success = True
    for fname, url in FILES.items():
        dest = DATA_DIR / fname
        if not download_file(url, dest, fname):
            success = False
            break

    # Try alternatives if primary failed
    if not success:
        success = try_alternative_sources()

    if success:
        print(f"\n  ✓ Download complete!")
        # Verify
        if npz_path.exists():
            import numpy as np
            d = np.load(npz_path)
            print(f"  ✓ pems04.npz shape: {d['data'].shape}")
    else:
        print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
