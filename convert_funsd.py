#!/usr/bin/env python3
"""
convert_label_studio_to_lilt.py
Convert Label Studio annotations to LiLT training format with relations.

✅ Extracts bounding boxes (HEADER, QUESTION, ANSWER)
✅ Extracts relations between tokens (from_id → to_id)
✅ Maps Label Studio IDs to token indices AFTER sorting
✅ Outputs JSON format compatible with LiLT training
✅ Handles multi-line answers (one QUESTION → multiple ANSWERs)
✅ Splits data into train/val directories for training
"""

import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse
import shutil
from datetime import datetime


class LabelStudioToLiLTConverter:
    """Convert Label Studio annotations to LiLT format with relations."""
    
    def __init__(self, images_dir: Path, output_dir: Path, val_ratio: float = 0.1, seed: int = 42):
        self.images_dir = Path(images_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.val_ratio = val_ratio
        self.seed = seed
        random.seed(seed)
        
        # Create train/val subdirectories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        for d in [self.train_dir, self.val_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Label mapping (Label Studio → LiLT)
        self.label_mapping = {
            'header': 'HEADER',
            'question': 'QUESTION',
            'answer': 'ANSWER',
            'other': 'O'
        }
        
        print(f"📁 Images directory: {self.images_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Train/Val split: {100*(1-val_ratio):.0f}% / {100*val_ratio:.0f}%")
    
    def find_image_file(self, task: dict) -> Optional[Path]:
        """Find the corresponding image file for a task."""
        # Try file_upload field first
        if 'file_upload' in task:
            filename = Path(task['file_upload']).name
            # Handle Label Studio's prefixed filenames like "abc123-image.png"
            if '-' in filename and len(filename.split('-')[0]) <= 10:
                # Likely a UUID prefix, extract actual filename
                parts = filename.split('-', 1)
                if len(parts) == 2:
                    clean_name = parts[1]
                else:
                    clean_name = filename
            else:
                clean_name = filename
            
            # Search for the file with various extensions
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf']:
                matches = list(self.images_dir.rglob(f"*{clean_name}"))
                if matches:
                    return matches[0].resolve()
            
            # Try exact match
            candidate = self.images_dir / filename
            if candidate.exists():
                return candidate.resolve()
        
        # Try data.image field
        if 'data' in task and 'image' in task['data']:
            image_path = task['data']['image']
            # Handle paths like "/data/upload/5/abc-image.png"
            filename = Path(image_path).name
            candidate = self.images_dir / filename
            if candidate.exists():
                return candidate.resolve()
        
        # Fallback: search by any image in task data
        if 'data' in task:
            for key, value in task['data'].items():
                if isinstance(value, str) and any(value.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf']):
                    filename = Path(value).name
                    candidate = self.images_dir / filename
                    if candidate.exists():
                        return candidate.resolve()
        
        return None
    
    def extract_bboxes_and_relations(self, annotation: dict, img_width: int, img_height: int) -> Tuple[List[dict], List[dict]]:
        """Extract bounding boxes and relations from an annotation."""
        results = annotation.get('result', [])
        
        bboxes = []
        relations_raw = []  # Store relations with original IDs first
        id_to_bbox_data = {}  # Map result ID to bbox data
        
        # First pass: collect all bounding boxes
        for result in results:
            if result.get('type') != 'rectanglelabels':
                continue
            
            value = result.get('value', {})
            labels = value.get('rectanglelabels', [])
            if not labels:
                continue
            
            x = value.get('x', 0)
            y = value.get('y', 0)
            width = value.get('width', 0)
            height = value.get('height', 0)
            
            # Convert to normalized coordinates (0-1000 scale for LiLT)
            x0 = max(0, min(1000, int((x / 100) * 1000)))
            y0 = max(0, min(1000, int((y / 100) * 1000)))
            x1 = max(0, min(1000, int(((x + width) / 100) * 1000)))
            y1 = max(0, min(1000, int(((y + height) / 100) * 1000)))
            
            raw_label = labels[0].lower()
            mapped_label = self.label_mapping.get(raw_label, 'O')
            
            # Store bbox with its original result ID for relation mapping
            bbox_data = {
                'result_id': result['id'],
                'label': mapped_label,
                'box': [x0, y0, x1, y1],
                'box_original': [x, y, width, height],
                'text': '',  # Will be filled by OCR during training
                'center_x': (x0 + x1) / 2,
                'center_y': (y0 + y1) / 2,
            }
            
            id_to_bbox_data[result['id']] = bbox_data
            bboxes.append(bbox_data)
        
        # Second pass: collect all relations (store with original IDs)
        for result in results:
            if result.get('type') != 'relation':
                continue
            
            from_id = result.get('from_id')
            to_id = result.get('to_id')
            direction = result.get('direction', 'right')
            rel_type = result.get('type', 'key_value')
            
            if from_id in id_to_bbox_data and to_id in id_to_bbox_data:
                relations_raw.append({
                    'from_result_id': from_id,
                    'to_result_id': to_id,
                    'type': rel_type,
                    'direction': direction
                })
        
        # Sort bboxes by reading order (top-to-bottom, left-to-right)
        bboxes.sort(key=lambda b: (b['center_y'], b['center_x']))
        
        # Create new mapping: result_id -> new index after sorting
        result_id_to_new_idx = {
            bbox['result_id']: idx for idx, bbox in enumerate(bboxes)
        }
        
        # Convert raw relations to use new indices
        relations = []
        for rel in relations_raw:
            from_idx = result_id_to_new_idx.get(rel['from_result_id'])
            to_idx = result_id_to_new_idx.get(rel['to_result_id'])
            
            if from_idx is not None and to_idx is not None:
                relations.append({
                    'head': from_idx,
                    'tail': to_idx,
                    'type': rel['type'],
                    'direction': rel['direction']
                })
        
        # Clean up bboxes: remove internal fields, keep only LiLT format
        clean_bboxes = []
        for idx, bbox in enumerate(bboxes):
            clean_bboxes.append({
                'id': idx,
                'label': bbox['label'],
                'box': bbox['box'],
                'text': bbox['text'],
                'center_x': bbox['center_x'],
                'center_y': bbox['center_y'],
            })
        
        return clean_bboxes, relations
    
    def convert_task(self, task: dict, task_idx: int) -> Optional[dict]:
        """Convert a single Label Studio task to LiLT format."""
        image_path = self.find_image_file(task)
        if not image_path:
            print(f"⚠️  Task {task_idx}: Image not found for {task.get('file_upload', 'N/A')}")
            return None
        
        annotations = task.get('annotations', [])
        if not annotations:
            print(f"⚠️  Task {task_idx}: No annotations")
            return None
        
        # Use first completed annotation (skip cancelled ones)
        annotation = None
        for ann in annotations:
            if not ann.get('was_cancelled', False):
                annotation = ann
                break
        
        if not annotation or not annotation.get('result'):
            print(f"⚠️  Task {task_idx}: No valid annotation results")
            return None
        
        # Get image dimensions from first result
        first_result = annotation['result'][0]
        img_width = first_result.get('original_width', 1655)
        img_height = first_result.get('original_height', 2339)
        
        bboxes, relations = self.extract_bboxes_and_relations(annotation, img_width, img_height)
        
        if not bboxes:
            print(f"⚠️  Task {task_idx}: No bounding boxes found")
            return None
        
        # Build LiLT format output
        lilt_format = {
            'id': task.get('id', task_idx),
            'image': image_path.name,
            'image_path': str(image_path.relative_to(self.images_dir.parent).as_posix()),
            'image_size': {
                'width': img_width,
                'height': img_height
            },
            'tokens': bboxes,
            'relations': relations,
            'stats': {
                'total_tokens': len(bboxes),
                'total_relations': len(relations),
                'header_count': sum(1 for b in bboxes if b['label'] == 'HEADER'),
                'question_count': sum(1 for b in bboxes if b['label'] == 'QUESTION'),
                'answer_count': sum(1 for b in bboxes if b['label'] == 'ANSWER'),
            }
        }
        
        return lilt_format
    
    def _save_split(self, tasks: List[dict], split_dir: Path, split_name: str):
        """Save a split of tasks to individual files and create index."""
        split_dir.mkdir(parents=True, exist_ok=True)
        index = []
        
        for task in tasks:
            # Save individual JSON file
            output_file = split_dir / f"{task['id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(task, f, indent=2, ensure_ascii=False)
            
            # Add to index
            index.append({
                'id': task['id'],
                'image': task['image'],
                'image_path': task['image_path'],
                'num_tokens': task['stats']['total_tokens'],
                'num_relations': task['stats']['total_relations']
            })
        
        # Save index file
        index_file = split_dir / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                'split': split_name,
                'count': len(index),
                'items': index
            }, f, indent=2, ensure_ascii=False)
        
        print(f"   📁 {split_name}: {len(tasks)} samples → {split_dir}/")
    
    def convert(self, export_file: Path) -> dict:
        """Convert entire Label Studio export to LiLT format with train/val split."""
        try:
            with open(export_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load export file: {e}")
            sys.exit(1)
        
        print(f"📄 Loaded {len(tasks)} tasks from {export_file}")
        
        # Convert all tasks first
        converted_tasks = []
        stats = {
            'total': len(tasks),
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'total_relations': 0,
            'label_distribution': defaultdict(int)
        }
        
        for idx, task in enumerate(tasks, 1):
            if idx % 10 == 0 or idx == len(tasks):
                print(f"\n🔄 Converting task {idx}/{len(tasks)} (ID: {task.get('id', 'N/A')})")
            
            result = self.convert_task(task, idx)
            if result:
                converted_tasks.append(result)
                stats['successful'] += 1
                stats['total_tokens'] += result['stats']['total_tokens']
                stats['total_relations'] += result['stats']['total_relations']
                
                for label in ['HEADER', 'QUESTION', 'ANSWER']:
                    stats['label_distribution'][label] += result['stats'][f'{label.lower()}_count']
            else:
                stats['failed'] += 1
        
        if not converted_tasks:
            print("❌ No tasks were successfully converted. Check your input files.")
            sys.exit(1)
        
        # Shuffle and split into train/val
        random.shuffle(converted_tasks)
        split_idx = int(len(converted_tasks) * (1 - self.val_ratio))
        train_tasks = converted_tasks[:split_idx]
        val_tasks = converted_tasks[split_idx:]
        
        print(f"\n📊 Split: {len(train_tasks)} train / {len(val_tasks)} val")
        
        # Save to train/val directories
        self._save_split(train_tasks, self.train_dir, "train")
        self._save_split(val_tasks, self.val_dir, "val")
        
        # Save combined metadata
        metadata = {
            'source': str(export_file),
            'converted_at': datetime.now().isoformat(),
            'label_mapping': self.label_mapping,
            'format': 'LiLT with relations',
            'split': {
                'train': len(train_tasks),
                'val': len(val_tasks),
                'val_ratio': self.val_ratio,
                'seed': self.seed
            },
            'stats': {
                'total_tokens': stats['total_tokens'],
                'total_relations': stats['total_relations'],
                'label_distribution': dict(stats['label_distribution'])
            }
        }
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*70)
        print("📊 CONVERSION SUMMARY")
        print("="*70)
        print(f"   Total tasks: {stats['total']}")
        print(f"   Successful: {stats['successful']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Total relations: {stats['total_relations']}")
        print(f"   Label distribution:")
        for label, count in sorted(stats['label_distribution'].items()):
            print(f"     - {label}: {count}")
        print(f"\n💾 Output structure:")
        print(f"   {self.output_dir}/")
        print(f"   ├── metadata.json")
        print(f"   ├── train/")
        print(f"   │   ├── index.json")
        print(f"   │   └── *.json (one per sample)")
        print(f"   └── val/")
        print(f"       ├── index.json")
        print(f"       └── *.json (one per sample)")
        print("="*70)
        
        return {
            'output_dir': str(self.output_dir),
            'train_count': len(train_tasks),
            'val_count': len(val_tasks),
            'stats': stats
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Label Studio annotations to LiLT training format with train/val split"
    )
    parser.add_argument("export_json", help="Label Studio export JSON file")
    parser.add_argument("images_dir", help="Directory containing invoice images")
    parser.add_argument("output_dir", help="Output directory for converted data (will contain train/ and val/)")
    parser.add_argument("--val-ratio", type=float, default=0.1, 
                        help="Validation split ratio (default: 0.1 = 10%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splitting (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    export_file = Path(args.export_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not export_file.exists():
        print(f"❌ Export file not found: {export_file}")
        sys.exit(1)
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not (0 < args.val_ratio < 1):
        print(f"❌ Validation ratio must be between 0 and 1 (got {args.val_ratio})")
        sys.exit(1)
    
    print("="*70)
    print("🔄 Label Studio → LiLT Converter (with Train/Val Split)")
    print("="*70)
    
    converter = LabelStudioToLiLTConverter(
        images_dir=images_dir,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    result = converter.convert(export_file)
    
    print("\n✅ Conversion complete!")
    print(f"📁 Training data: {output_dir}/train/")
    print(f"📁 Validation data: {output_dir}/val/")
    print(f"\n💡 To train LiLT model:")
    print(f"   python train.py --data_dir {output_dir}/train --val_dir {output_dir}/val")


if __name__ == "__main__":
    main()