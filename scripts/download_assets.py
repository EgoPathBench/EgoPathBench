"""
Download all asset_library files from InternScenes to /datadisk.
Downloads in order from smallest to largest, with progress reporting.
"""
import os
import sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = "InternRobotics/InternScenes"
LOCAL_DIR = "/datadisk/NavBench3D/internscenes_raw"

api = HfApi()


def download_flat_files(subpath: str, label: str):
    """Download all files directly under asset_library/{subpath}/."""
    print(f"\n{'='*60}")
    print(f"Downloading {label}: asset_library/{subpath}")
    print(f"{'='*60}")
    items = list(api.list_repo_tree(REPO, repo_type="dataset",
                                     path_in_repo=f"asset_library/{subpath}", recursive=True))
    files = [i for i in items if hasattr(i, "size") and i.size]
    total_size = sum(f.size for f in files)
    print(f"  {len(files)} files, {total_size/1e9:.2f} GB total")

    downloaded = 0
    for i, f in enumerate(files, 1):
        local_path = os.path.join(LOCAL_DIR, f.path)
        if os.path.exists(local_path):
            downloaded += f.size
            continue
        hf_hub_download(REPO, repo_type="dataset", filename=f.path, local_dir=LOCAL_DIR)
        downloaded += f.size
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {downloaded/1e9:.1f}/{total_size/1e9:.1f} GB")
    print(f"  Done: {label}")


def download_folder_recursive(subpath: str, label: str):
    """Download an entire folder recursively."""
    print(f"\n{'='*60}")
    print(f"Downloading {label}: asset_library/{subpath}")
    print(f"{'='*60}")
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=[f"asset_library/{subpath}/**"],
        max_workers=4,
    )
    print(f"  Done: {label}")


if __name__ == "__main__":
    # 1. gen_assets (tiny, ~5 dirs)
    download_folder_recursive("gen_assets", "gen_assets (tiny)")

    # 2. hssd-models (dir structure)
    download_folder_recursive("hssd-models", "hssd-models")

    # 3. 3D-FUTURE-model (~12.7GB, flat .glb files)
    download_flat_files("3D-FUTURE-model", "3D-FUTURE-model (~12.7GB)")

    # 4. gr100 (~34GB, flat .glb files)
    download_flat_files("gr100", "gr100 (~34GB)")

    # 5. objaverse_old (~42GB, flat .glb files)
    download_flat_files("objaverse_old", "objaverse_old (~42GB)")

    # 6. objaverse (~102GB, split tar.gz)
    download_flat_files("objaverse", "objaverse (~102GB split tar)")

    print("\n" + "="*60)
    print("ALL ASSET LIBRARIES DOWNLOADED!")
    print("="*60)
