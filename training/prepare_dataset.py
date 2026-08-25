"""
Dataset Preparation Script for Plant Disease Detection
------------------------------------------------------
This script splits raw PlantVillage dataset images into Train (70%), 
Validation (15%), and Test (15%) sets while preserving class folder structures.

INSTRUCTIONS:
1. Download the PlantVillage dataset from Kaggle or GitHub.
2. Extract the dataset folder (containing class folders like 'Tomato___Bacterial_spot', etc.)
   into the following directory:
   
   plant-disease-ai/dataset/raw/
   
   Example structure before running this script:
   plant-disease-ai/dataset/raw/
   ├── Apple___Apple_scab/
   ├── Apple___Black_rot/
   ├── Tomato___Bacterial_spot/
   └── ...

3. Run this script from the project root (`plant-disease-ai` directory):
   python training/prepare_dataset.py
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# Supported image file extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def prepare_dataset(source_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits dataset from source_dir into train, validation, and test folders in output_dir.
    """
    random.seed(seed)
    
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    if not source_path.exists():
        print(f"\n[ERROR] Source dataset directory not found at:\n  {source_path}\n")
        print("Please place your extracted PlantVillage dataset into the directory above.")
        print("Expected structure:")
        print("  dataset/raw/")
        print("  ├── Class1_Folder/")
        print("  ├── Class2_Folder/")
        print("  └── ...\n")
        return

    # Automatically detect all class subdirectories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"\n[ERROR] No subdirectories (classes) found in:\n  {source_path}\n")
        print("Please ensure your dataset directory contains folders for each plant disease class.")
        return

    train_path = output_path / "train"
    val_path = output_path / "validation"
    test_path = output_path / "test"

    print("=" * 60)
    print("PLANT DISEASE DATASET PREPARATION")
    print("=" * 60)
    print(f"Source Directory     : {source_path}")
    print(f"Output Directory     : {output_path}")
    print(f"Detected Classes     : {len(class_dirs)}")
    print(f"Split Ratios         : Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    print("=" * 60)

    total_train_count = 0
    total_val_count = 0
    total_test_count = 0

    summary_table = []

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Get all valid image files in class folder
        image_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        
        if not image_files:
            print(f"[WARNING] Skipping empty or non-image class folder: {class_name}")
            continue

        # Shuffle images deterministically for reproducible split
        random.shuffle(image_files)

        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        # Test gets the remainder to avoid losing images due to rounding
        test_count = total_images - train_count - val_count

        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]

        # Create destination class subdirectories
        train_class_dir = train_path / class_name
        val_class_dir = val_path / class_name
        test_class_dir = test_path / class_name

        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy images (preserves original files without permanent resizing)
        for img in train_files:
            shutil.copy2(img, train_class_dir / img.name)
            
        for img in val_files:
            shutil.copy2(img, val_class_dir / img.name)
            
        for img in test_files:
            shutil.copy2(img, test_class_dir / img.name)

        total_train_count += len(train_files)
        total_val_count += len(val_files)
        total_test_count += len(test_files)

        summary_table.append((class_name, total_images, len(train_files), len(val_files), len(test_files)))

    print("\nCLASS-BY-CLASS SUMMARY:")
    print(f"{'Class Name':<45} | {'Total':<7} | {'Train':<7} | {'Val':<7} | {'Test':<7}")
    print("-" * 80)
    for class_name, tot, tr, val, tst in summary_table:
        print(f"{class_name:<45} | {tot:<7} | {tr:<7} | {val:<7} | {tst:<7}")
    
    print("-" * 80)
    print(f"{'TOTAL SUMMARY':<45} | {total_train_count+total_val_count+total_test_count:<7} | {total_train_count:<7} | {total_val_count:<7} | {total_test_count:<7}")
    print("=" * 60)
    print("\nDataset preparation completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split PlantVillage dataset into train/validation/test folders.")
    parser.add_argument('--source', type=str, default='dataset/raw', help='Path to raw PlantVillage dataset directory.')
    parser.add_argument('--output', type=str, default='dataset', help='Path to output dataset directory.')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='Train split ratio (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test split ratio (default: 0.15)')

    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
