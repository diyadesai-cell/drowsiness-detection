import os
import hashlib

# ---------------------------
# Config
# ---------------------------
train_dir = r"C:\Users\91973\Desktop\drowsiness dataset2\train_data"
val_dir   = r"C:\Users\91973\Desktop\drowsiness dataset2\val_data"

def get_file_hash(filepath):
    """Return MD5 hash of a file (to detect duplicates)"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def scan_folder(folder):
    """Return dictionary {hash: filepath} for all files"""
    file_hashes = {}
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            file_hashes[get_file_hash(path)] = path
    return file_hashes

# ---------------------------
# Scan train & val folders
# ---------------------------
print("Scanning train and val folders...")
train_hashes = scan_folder(train_dir)
val_hashes   = scan_folder(val_dir)

# ---------------------------
# Check duplicates
# ---------------------------
duplicates = set(train_hashes.keys()) & set(val_hashes.keys())

if duplicates:
    print(f"\n Found {len(duplicates)} duplicate images between train and val!")
    for h in list(duplicates)[:10]:  # show first 10 duplicates
        print(f"Train: {train_hashes[h]}")
        print(f"Val  : {val_hashes[h]}")
        print("---")
else:
    print("\n No duplicates found between train and val.")
