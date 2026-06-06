import sys
from pathlib import Path

# Add src/ to python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from weird_hazelnut.config import load_config
from weird_hazelnut.data import create_data_layer
from weird_hazelnut.data.repositories import DataRepository

def main():
    config = load_config()
    # Connect using active config (config.docker.yaml inside container)
    data_layer = create_data_layer(config)
    if not data_layer:
        print("Failed to initialize Data Layer. Verify data_layer.enabled is true in config.")
        return
        
    with data_layer.database.session() as session:
        repo = DataRepository(session)
        
        # Group and list
        for split in ["train", "val", "test"]:
            items = repo.list_dataset_items(split=split)
            print(f"\n=================== SPLIT: {split.upper()} (Total: {len(items)}) ===================")
            
            # Count by label
            counts = {}
            for item, img in items:
                counts[item.label] = counts.get(item.label, 0) + 1
            print("Label counts:", counts)
            
            # Show detailed list of items
            print("\nDetailed samples (first 30):")
            for idx, (item, img) in enumerate(items[:30]):
                print(f"  {idx+1:02d}. Image ID: {img.id:<36} | Label: {item.label:<10} | File: {img.original_filename or 'None':<30}")
            if len(items) > 30:
                print(f"  ... and {len(items) - 30} more items.")

if __name__ == "__main__":
    main()
