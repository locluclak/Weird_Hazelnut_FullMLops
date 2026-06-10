import os
import re
import sys
import shutil
import urllib.parse
from pathlib import Path

# Add src/ to python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from label_studio_sdk.client import LabelStudio

def main():
    # Load API Key from .env
    env_path = Path(".env")
    api_key = None
    url = "http://localhost:8080"
    project_id = 1

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("LABEL_STUDIO_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                elif line.startswith("LABEL_STUDIO_URL="):
                    url = line.split("=", 1)[1].strip()
                elif line.startswith("LABEL_STUDIO_PROJECT_ID="):
                    project_id = int(line.split("=", 1)[1].strip())
                    
    # Fallback to internal docker container name for connecting inside docker container
    url = "http://label-studio:8080"

    if not api_key:
        print("LABEL_STUDIO_API_KEY not found in .env!")
        return

    print(f"Connecting to Label Studio at {url}...")
    client = LabelStudio(base_url=url, api_key=api_key)
    
    print(f"Fetching tasks for project {project_id}...")
    try:
        tasks = client.tasks.list(project=project_id)
    except Exception as e:
        print(f"Failed to list tasks: {e}")
        return

    # Serve data/lake at data/lake
    lake_root = Path("data/lake")
    lake_root.mkdir(parents=True, exist_ok=True)

    repaired_count = 0
    for task in tasks:
        image_val = task.data.get("image", "")
        if not image_val:
            continue
            
        # Check if URL is local /static/...
        if "/static/" in image_val:
            path_part = image_val.split("/static/", 1)[1]
            decoded_path = urllib.parse.unquote(path_part)
            target_path = lake_root / decoded_path
            
            if target_path.exists():
                continue
                
            # Restore/repair missing file
            print(f"Task {task.id} image is missing: {target_path}")
            filename = target_path.name.lower()
            defect_class = None
            for c in ["crack", "cut", "hole", "print"]:
                if f"_{c}.png" in filename:
                    defect_class = c
                    break
                    
            if not defect_class:
                defect_class = "good"
                
            src_dir = Path("data/hazelnut/test") / defect_class
            if not src_dir.exists():
                src_dir = Path("data/hazelnut/train/good")
                
            src_images = sorted(list(src_dir.glob("*.png")))
            if not src_images:
                continue
                
            src_img = src_images[0]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, target_path)
            print(f"  ✅ Restored image of class '{defect_class}' -> {target_path}")
            repaired_count += 1

    print(f"\nRepair session completed! Restored {repaired_count} missing static images.")

if __name__ == "__main__":
    main()
