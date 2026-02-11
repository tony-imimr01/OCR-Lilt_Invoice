#!/usr/bin/env python3
"""
png_to_funsd_bio_fixed.py — Handles B-/I- tags for LILT model training
FIXED: Properly implements BIO (IOB) tagging scheme with B- and I- tags
"""
import json
import sys
from pathlib import Path
from urllib.parse import unquote
from PIL import Image
import argparse
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BioConverter:
    def __init__(self, images_dir: Path, output_dir: Path):
        self.images_dir = Path(images_dir).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_mapping = {
            'header': 'HEADER',
            'question': 'QUESTION',
            'answer': 'ANSWER',
            'other': 'OTHER'
        }
    
    def find_image_path(self, task: dict) -> str:
        """Robustly find image path from multiple possible fields"""
        for field in ['image', 'file', 'url', 'image_url']:
            if 'data' in task and field in task['data']:
                return task['data'][field]
            if field in task:
                return task[field]
        return ""
    
    def find_image_file(self, image_path: str) -> Path:
        """Resolve image path to actual file"""
        if not image_path:
            return None
        
        filename = Path(unquote(image_path)).name
        
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
            candidate = self.images_dir / filename.replace('.pdf', ext).replace('.PDF', ext)
            if candidate.exists():
                return candidate
        
        # Recursive search as fallback
        for ext in ['.png', '.jpg', '.jpeg']:
            matches = list(self.images_dir.rglob(f"*{Path(filename).stem}*{ext}"))
            if matches:
                return matches[0]
        
        return None
    
    def normalize_bbox(self, x_pct, y_pct, w_pct, h_pct, img_w, img_h):
        x0 = max(0, min(int(1000 * x_pct / 100), 1000))
        y0 = max(0, min(int(1000 * y_pct / 100), 1000))
        x1 = max(x0 + 1, min(int(1000 * (x_pct + w_pct) / 100), 1000))
        y1 = max(y0 + 1, min(int(1000 * (y_pct + h_pct) / 100), 1000))
        return [x0, y0, x1, y1]
    
    def split_text_to_tokens(self, text: str, bbox: list, label_type: str):
        """Split text into tokens and assign BIO tags"""
        tokens = []
        words = text.split()
        
        if not words:
            return tokens
        
        # Calculate per-word bbox approximation
        x0, y0, x1, y1 = bbox
        total_width = x1 - x0
        char_count = len(text.replace(' ', ''))
        
        if char_count == 0:
            return tokens
        
        char_width = total_width / char_count if char_count > 0 else 0
        current_x = x0
        
        for i, word in enumerate(words):
            # Estimate word width based on character count
            word_chars = len(word)
            word_width = word_chars * char_width
            
            # Create word bbox
            word_bbox = [
                int(current_x),
                y0,
                int(current_x + word_width),
                y1
            ]
            
            # Clamp values
            word_bbox[0] = max(x0, min(word_bbox[0], x1))
            word_bbox[2] = max(word_bbox[0] + 1, min(word_bbox[2], x1))
            
            # Assign BIO tag
            if i == 0:  # First word gets B- tag
                bio_tag = f"B-{label_type}"
            else:       # Subsequent words get I- tag
                bio_tag = f"I-{label_type}"
            
            tokens.append({
                "text": word,
                "box": word_bbox,
                "label": bio_tag
            })
            
            # Update position for next word (add space width)
            current_x += word_width + (char_width if char_width > 0 else 5)
        
        return tokens
    
    def convert_task(self, task: dict, idx: int) -> bool:
        # Get image path
        image_path = self.find_image_path(task)
        if not image_path:
            logger.warning(f"⚠️ Task {idx}: Missing image field")
            return False
        
        # Find actual image file
        img_file = self.find_image_file(image_path)
        if not img_file:
            logger.warning(f"⚠️ Task {idx}: Image not found for '{image_path}'")
            return False
        
        # Get annotations
        annotations = task.get('annotations', [])
        if not annotations or not annotations[0].get('result'):
            logger.error(f"🚨 Task {idx}: EMPTY ANNOTATIONS!")
            return False
        
        results = annotations[0]['result']
        all_tokens = []
        
        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except Exception as e:
            logger.error(f"❌ Task {idx}: Failed to open image {img_file}: {e}")
            return False
        
        # Process each annotation result
        for result in results:
            if result.get('type') != 'rectanglelabels':
                continue
            
            value = result.get('value', {})
            labels = value.get('rectanglelabels', [])
            if not labels:
                continue
            
            # Get bbox
            x = value.get('x', 0)
            y = value.get('y', 0)
            w = value.get('width', 0)
            h = value.get('height', 0)
            if w <= 0 or h <= 0:
                continue
            
            # Convert to normalized bbox
            norm_bbox = self.normalize_bbox(x, y, w, h, width, height)
            
            # Get text and label
            text = value.get('text', [''])[0].strip()
            if not text:
                # If no text provided, use label as placeholder
                text = f"[{labels[0]}]"
            
            # Get base label type
            base_label = self.label_mapping.get(labels[0].lower(), 'OTHER')
            
            # Split text into tokens with BIO tags
            tokens = self.split_text_to_tokens(text, norm_bbox, base_label)
            
            # Add tokens to the list
            all_tokens.extend(tokens)
        
        if not all_tokens:
            logger.warning(f"⚠️ Task {idx}: No valid tokens extracted")
            return False
        
        # Create entities by grouping consecutive tokens with same entity type
        entities = []
        current_entity = None
        
        for token in all_tokens:
            label = token['label']
            base_type = label[2:]  # Remove B- or I- prefix
            
            if label.startswith('B-'):
                # Start a new entity
                if current_entity is not None:
                    entities.append(current_entity)
                
                current_entity = {
                    "id": len(entities),
                    "text": token["text"],
                    "label": base_type,
                    "words": [{"box": token["box"], "text": token["text"]}],
                    "box": token["box"].copy(),  # Will expand as we add more words
                    "linking": []
                }
            elif label.startswith('I-') and current_entity is not None and current_entity["label"] == base_type:
                # Continue current entity
                current_entity["text"] += " " + token["text"]
                current_entity["words"].append({"box": token["box"], "text": token["text"]})
                # Expand entity bbox to include this token
                current_entity["box"][0] = min(current_entity["box"][0], token["box"][0])
                current_entity["box"][1] = min(current_entity["box"][1], token["box"][1])
                current_entity["box"][2] = max(current_entity["box"][2], token["box"][2])
                current_entity["box"][3] = max(current_entity["box"][3], token["box"][3])
            else:
                # Shouldn't happen if BIO tags are correct
                logger.warning(f"⚠️ Task {idx}: Invalid BIO sequence")
        
        # Add the last entity if exists
        if current_entity is not None:
            entities.append(current_entity)
        
        # Build FUNSD doc
        doc_id = f"invoice_{idx}_{img_file.stem}"
        funsd_doc = {
            "id": doc_id,
            "file": img_file.name,
            "img": {"fname": img_file.name, "width": width, "height": height},
            "document": entities,
            "label": 0
        }
        
        output = {
            "lang": "en",
            "version": "1.0",
            "split": "train",
            "label": 0,
            "documents": [funsd_doc]
        }
        
        # Save JSON
        output_path = self.output_dir / f"{img_file.stem}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Copy image to output directory
        img_dest = self.output_dir / img_file.name
        if not img_dest.exists():
            try:
                shutil.copy2(img_file, img_dest)
            except Exception as e:
                logger.warning(f"⚠️ Failed to copy image: {e}")
        
        # Count BIO tags for logging
        b_count = sum(1 for t in all_tokens if t['label'].startswith('B-'))
        i_count = sum(1 for t in all_tokens if t['label'].startswith('I-'))
        
        logger.info(f"✅ Task {idx}: '{img_file.name}' → {len(entities)} entities, {len(all_tokens)} tokens (B:{b_count}, I:{i_count})")
        return True
    
    def convert(self, export_file: Path):
        # Handle BOM in Label Studio exports
        try:
            with open(export_file, encoding='utf-8-sig') as f:
                tasks = json.load(f)
        except UnicodeDecodeError:
            with open(export_file, encoding='utf-8') as f:
                tasks = json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load export file: {e}")
            sys.exit(1)
        
        logger.info(f"✅ Loaded {len(tasks)} tasks from {export_file}")
        
        success = 0
        for idx, task in enumerate(tasks, 1):
            if self.convert_task(task, idx):
                success += 1
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ CONVERSION COMPLETE: {success}/{len(tasks)} tasks converted")
        logger.info(f"📊 Label mapping: {self.label_mapping}")
        logger.info(f"🏷️  Using BIO scheme: B- for first token, I- for subsequent tokens")
        logger.info(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Convert Label Studio exports to FUNSD format with BIO tags")
    parser.add_argument("export_json", help="Label Studio export JSON")
    parser.add_argument("images_dir", help="Directory with PNG/JPG images")
    parser.add_argument("output_dir", help="Output directory for FUNSD format")
    args = parser.parse_args()
    
    export_file = Path(args.export_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    if not export_file.exists():
        logger.error(f"❌ Export file not found: {export_file}")
        sys.exit(1)
    
    if not images_dir.exists():
        logger.error(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)
    
    converter = BioConverter(
        images_dir=images_dir,
        output_dir=output_dir
    )
    converter.convert(export_file)


if __name__ == "__main__":
    main()