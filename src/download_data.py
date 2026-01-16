import os
import sys
import subprocess
import zipfile
from pathlib import Path

# Kaggle dataset
DATASET_SLUG = "cryptexcode/banfakenews"

# Target directory
RAW_DATA_DIR = Path("data") / "raw"

# Expected final filenames (used by project)
EXPECTED_FILES = {
    "Authentic-48.csv": ["authentic-48", "authentic"],
    "Fake-1K.csv": ["fake-1k", "fake"],
    "LabeledAuthentic-7K.csv": ["labeledauthentic", "labeled_authentic"],
    "LabeledFake-1K.csv": ["labeledfake", "labeled_fake"],
}


def check_kaggle_cli():
    """Ensure Kaggle CLI is installed and accessible."""
    try:
        subprocess.run(
            ["kaggle", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("ERROR: Kaggle CLI not found.")
        print("Install it with: pip install kaggle")
        print("Then create API token at: https://www.kaggle.com/account")
        sys.exit(1)


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kaggle dataset: {DATASET_SLUG}")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(RAW_DATA_DIR),
            "--force",
        ],
        check=True,
    )

    zip_files = list(RAW_DATA_DIR.glob("*.zip"))
    if not zip_files:
        print("ERROR: No ZIP file downloaded.")
        sys.exit(1)

    return zip_files[0]


def extract_dataset(zip_path: Path):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DATA_DIR)
    zip_path.unlink()  # remove zip after extraction


def normalize_filenames():
    csv_files = list(RAW_DATA_DIR.rglob("*.csv"))
    if not csv_files:
        print("ERROR: No CSV files found after extraction.")
        sys.exit(1)

    for expected, keywords in EXPECTED_FILES.items():
        matched = False
        for csv in csv_files:
            name = csv.name.lower().replace(" ", "").replace("-", "")
            if any(k in name for k in keywords):
                target = RAW_DATA_DIR / expected
                if csv.resolve() != target.resolve():
                    csv.rename(target)
                print(f"✔ {expected}")
                matched = True
                break

        if not matched:
            print(f"ERROR: Required file not found for {expected}")
            sys.exit(1)


def verify():
    print("\nFinal verification:")
    for fname in EXPECTED_FILES:
        path = RAW_DATA_DIR / fname
        if not path.exists():
            print(f"❌ Missing: {fname}")
            sys.exit(1)
        else:
            print(f"✅ {fname}")

    print("\nDataset is ready.")


if __name__ == "__main__":
    check_kaggle_cli()
    zip_path = download_dataset()
    extract_dataset(zip_path)
    normalize_filenames()
    verify()
