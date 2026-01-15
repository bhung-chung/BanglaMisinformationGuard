import os
import urllib.request
import zipfile
import shutil
import io

RAW_DATA_DIR = os.path.join("data", "raw")
REPO_ZIP_URLS = [
    "https://github.com/Rowan1224/FakeNews/archive/refs/heads/master.zip",
    "https://github.com/Rowan1224/FakeNews/archive/refs/heads/main.zip"
]
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def download_and_extract():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    
    success = False
    for url in REPO_ZIP_URLS:
        print(f"Trying to download ZIP from {url}...")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req) as response:
                zip_content = response.read()
                
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
                # List files to find where the CSVs are
                print("Files in ZIP:")
                for file in zip_ref.namelist():
                    # Check if it looks like our data
                    if file.endswith('.csv') and 'Authentic' in file:
                        print(f"  Found: {file}")
                        # Extract flat to RAW_DATA_DIR
                        filename = os.path.basename(file)
                        target_path = os.path.join(RAW_DATA_DIR, filename)
                        with zip_ref.open(file) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        print(f"  Extracted to {target_path}")
                        success = True
                    elif file.endswith('.csv') and 'Fake' in file:
                         filename = os.path.basename(file)
                         target_path = os.path.join(RAW_DATA_DIR, filename)
                         with zip_ref.open(file) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                         print(f"  Extracted to {target_path}")
            
            if success:
                print("Successfully extracted data.")
                break
        except Exception as e:
            print(f"Failed {url}: {e}")

    if not success:
        print("Failed to download data from any source.")

if __name__ == "__main__":
    download_and_extract()
