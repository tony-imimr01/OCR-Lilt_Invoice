#!/usr/bin/env python3
"""
split_dataset_fixed.py — Robust train/val split for LiLT datasets
FIXES:
✅ Handles images in SAME directory as JSON (not 'images' subfolder)
✅ Validates input directory exists and contains data
✅ Provides actionable error messages
✅ Handles edge cases (small datasets, missing images)
"""
import json
import shutil
from pathlib import Path
import random
import sys

def split_dataset(input_dir, train_dir, val_dir, train_ratio=0.8):
    input_dir = Path(input_dir).resolve()
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    
    # VALIDATION: Check input directory exists
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        print(f"   Did you mean: {input_dir.parent / 'lilt_dataset3'} ?")
        sys.exit(1)
    
    # VALIDATION: Check for JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"❌ ERROR: No JSON files found in {input_dir}")
        print(f"   Expected files like: invoice_001.json")
        print(f"\n💡 TROUBLESHOOTING:")
        print(f"   1. Run converter first: python convert_funsd_robust.py ...")
        print(f"   2. Verify converter output directory: {input_dir}")
        print(f"   3. List contents: ls {input_dir}")
        sys.exit(1)
    
    # VALIDATION: Check for images
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(input_dir.glob(f"*{ext}"))
    
    if not image_files:
        print(f"⚠️ WARNING: No images found in {input_dir}")
        print(f"   Converter should have saved images alongside JSON files")
        print(f"   Expected files like: invoice_001.png")
    
    print(f"✅ Found {len(json_files)} JSON files and {len(image_files)} images in {input_dir}")
    
    # Create output directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(json_files)
    
    split_idx = int(len(json_files) * train_ratio)
    train_files = json_files[:split_idx] if split_idx > 0 else json_files[:1]
    val_files = json_files[split_idx:] if split_idx < len(json_files) else json_files[-1:]
    
    # Copy train files
    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
        # Find matching image (same stem, any extension)
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            img_path = f.with_suffix(ext)
            if img_path.exists():
                shutil.copy2(img_path, train_dir / img_path.name)
                break
    
    # Copy val files
    for f in val_files:
        shutil.copy2(f, val_dir / f.name)
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            img_path = f.with_suffix(ext)
            if img_path.exists():
                shutil.copy2(img_path, val_dir / img_path.name)
                break
    
    print(f"\n✅ SPLIT COMPLETE")
    print(f"   Train: {len(train_files)} files → {train_dir}")
    print(f"   Val:   {len(val_files)} files → {val_dir}")
    print(f"\n💡 NEXT STEP: Train model with:")
    print(f"   python train_lilt_escape_o_trap.py \\")
    print(f"     --train_dir {train_dir} \\")
    print(f"     --val_dir {val_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset into train/val sets")
    parser.add_argument("input_dir", help="Input directory with JSON + images")
    parser.add_argument("train_dir", help="Output train directory")
    parser.add_argument("val_dir", help="Output val directory")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    args = parser.parse_args()
    
    split_dataset(args.input_dir, args.train_dir, args.val_dir, args.ratio)


if __name__ == "__main__":
    # EXAMPLE USAGE (FIXED PATHS):
    # split_dataset(
    #     "/home/tony/Desktop/LiLT_Invoice/lilt_dataset3",  # ← CORRECTED: "dataset3" not "datase3"
    #     "/home/tony/Desktop/LiLT_Invoice/lilt_dataset3/train",
    #     "/home/tony/Desktop/LiLT_Invoice/lilt_dataset3/val",
    #     train_ratio=0.8
    # )
    
    # Or run with CLI args:
    # python split_dataset_fixed.py lilt_dataset3 lilt_dataset3/train lilt_dataset3/val --ratio 0.8
    
    main()