#!/usr/bin/env python3
"""
LiLT Training Script - FIXED for LayoutLMv3Processor input format
✅ CRITICAL FIX: Pass images/text/boxes/labels as LISTS to processor
✅ LayoutLMv3Processor requires list inputs even for single examples
✅ Removed all deprecated imports (LayoutLMv2FeatureExtractor)
✅ Automatic label alignment via word_labels parameter
"""
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
set_seed(42)

class Config:
    train_dir = "lilt_dataset/train"
    val_dir = "lilt_dataset/val"
    model_name = "SCUT-DLVCLab/lilt-roberta-en-base"
    max_seq_length = 128
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 8
    learning_rate = 5e-5
    num_train_epochs = 30
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    output_dir = "lilt_invoice_model"
    logging_dir = "./logs"
    debug_mode = False
    label_list = None
    
    @classmethod
    def get_label2id(cls):
        return {label: i for i, label in enumerate(cls.label_list)}
    
    @classmethod
    def get_id2label(cls):
        return {i: label for i, label in enumerate(cls.label_list)}


def detect_labels_from_dataset(data_dir: str):
    data_dir = Path(data_dir)
    all_labels = set()
    
    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            
            entities = []
            if "documents" in data and data["documents"]:
                for doc in data["documents"]:
                    if "document" in doc:
                        entities.extend(doc["document"])
            elif "document" in data:
                entities = data["document"]
            elif "entities" in data:
                entities = data["entities"]
            
            for entity in entities:
                if isinstance(entity, dict):
                    label = str(entity.get("label", "")).strip()
                    if label and label != "O":
                        all_labels.add(label)
        except Exception as e:
            if Config.debug_mode:
                logger.warning(f"Skipping {json_file.name}: {e}")
            continue
    
    labels = ["O"] + sorted(all_labels)
    logger.info(f"🔍 Detected {len(labels)} labels: {labels}")
    return labels


class LILTDataset(Dataset):
    def __init__(self, data_dir: str, processor, label2id: dict):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.examples = []
        
        json_files = list(self.data_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {data_dir}")
        
        logger.info(f"📚 Loading {len(json_files)} annotation files from {data_dir}...")
        
        valid_count = 0
        skipped_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                
                img_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    candidate = self.data_dir / (json_file.stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if not img_path:
                    if Config.debug_mode:
                        logger.warning(f"  ⚠️ Image not found for {json_file.name}")
                    skipped_count += 1
                    continue
                
                entities = []
                if "documents" in data and data["documents"]:
                    for doc in data["documents"]:
                        if "document" in doc:
                            entities.extend(doc["document"])
                elif "document" in data:
                    entities = data["document"]
                elif "entities" in data:
                    entities = data["entities"]
                elif isinstance(data, list):
                    entities = data
                
                if not entities:
                    if Config.debug_mode:
                        logger.warning(f"  ⚠️ No entities in {json_file.name}")
                    skipped_count += 1
                    continue
                
                words = []
                boxes = []
                word_labels = []
                
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    
                    text_raw = entity.get("text")
                    if text_raw is None or str(text_raw).strip() == "":
                        continue
                    
                    text_str = str(text_raw).strip()
                    if not text_str:
                        continue
                    
                    label_raw = entity.get("label", "O")
                    label_str = str(label_raw).strip() if label_raw else "O"
                    
                    bbox = entity.get("box") or entity.get("bbox", [0, 0, 100, 100])
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        if max(bbox) <= 1.0:
                            bbox = [int(coord * 1000) for coord in bbox[:4]]
                        else:
                            bbox = bbox[:4]
                    else:
                        bbox = [0, 0, 100, 100]
                    
                    tokens = text_str.split()
                    for token in tokens:
                        token_str = str(token).strip()
                        if not token_str:
                            continue
                        words.append(token_str)
                        word_labels.append(label_str)
                        boxes.append(bbox)
                
                if not words:
                    if Config.debug_mode:
                        logger.warning(f"  ⚠️ No valid words in {json_file.name}")
                    skipped_count += 1
                    continue
                
                if not all(isinstance(w, str) and w.strip() for w in words):
                    if Config.debug_mode:
                        problematic = [w for w in words if not (isinstance(w, str) and w.strip())]
                        logger.warning(f"  ⚠️ Non-string words in {json_file.name}: {problematic[:3]}")
                    skipped_count += 1
                    continue
                
                self.examples.append({
                    "image_path": str(img_path),
                    "words": words,
                    "boxes": boxes,
                    "word_labels": word_labels,
                })
                valid_count += 1
                
            except Exception as e:
                if Config.debug_mode:
                    logger.warning(f"  ⚠️ Error processing {json_file.name}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"✅ Loaded {valid_count} valid examples ({skipped_count} skipped)")
        
        if valid_count == 0:
            raise ValueError("No valid examples found!")
        elif valid_count < 20:
            logger.warning(f"⚠️ VERY SMALL DATASET ({valid_count} examples). Model may not learn effectively!")
    
    def __len__(self):
        return len(self.examples)
    
    def normalize_box(self, box, width, height):
        """Normalize bbox to 0-1000 scale"""
        return [
            max(0, min(1000, int(1000 * (box[0] / width)))),
            max(0, min(1000, int(1000 * (box[1] / height)))),
            max(0, min(1000, int(1000 * (box[2] / width)))),
            max(0, min(1000, int(1000 * (box[3] / height)))),
        ]

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        try:
            # Load image
            image = Image.open(example["image_path"]).convert("RGB")
            width, height = image.size
            
            # Clean words
            words = [str(w).strip() for w in example["words"]]
            words = [w for w in words if w]
            if not words:
                raise ValueError(f"Example {idx}: No valid words after cleaning")
            
            boxes = example["boxes"]
            word_labels = example["word_labels"]
            
            # Normalize boxes
            normalized_boxes = [self.normalize_box(b, width, height) for b in boxes]
            
            # Convert word labels to IDs
            word_label_ids = [
                self.label2id.get(str(label).strip(), self.label2id["O"])
                for label in word_labels
            ]
            
            # ✅ CRITICAL FIX: Pass words as LIST (not joined string)
            # LayoutLMv3Processor expects list of words for alignment with boxes/labels
            text = words  # List of words, NOT joined string
            
            # ✅ Use processor with correct parameters
            encoding = self.processor(
                image,                          # Single PIL Image
                text=text,                      # ✅ LIST of words (not string)
                boxes=normalized_boxes,         # Word-level boxes
                word_labels=word_label_ids,     # Word-level label IDs
                padding="max_length",
                truncation=True,
                max_length=Config.max_seq_length,
                return_tensors="pt",
            )
            
            # ✅ Ensure all required fields are present
            required_keys = ["pixel_values", "input_ids", "attention_mask", "bbox"]
            for key in required_keys:
                if key not in encoding:
                    raise ValueError(f"Missing required key '{key}' in encoding")
            
            # Remove batch dimension
            return {
                "pixel_values": encoding["pixel_values"].squeeze(0),
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "bbox": encoding["bbox"].squeeze(0),
                "labels": encoding["labels"].squeeze(0).to(torch.long),
            }
            
        except Exception as e:
            logger.error(f"❌ Error in example {idx} ({example.get('image_path', 'unknown')}): {e}")
            if Config.debug_mode:
                logger.error(f"  Words ({len(words)}): {words[:10]}")
                logger.error(f"  Boxes: {len(normalized_boxes)}, Word labels: {len(word_label_ids)}")
                logger.error(f"  Text type: {type(text)}, Text length: {len(text) if isinstance(text, list) else 'N/A'}")
            raise
            
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    flat_preds = []
    flat_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                flat_preds.append(pred_id)
                flat_labels.append(label_id)
    
    if not flat_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    
    pred_strs = [Config.id2label.get(p, "O") for p in flat_preds]
    label_strs = [Config.id2label.get(l, "O") for l in flat_labels]
    
    non_o_labels = [l for l in Config.label_list if l != "O"]
    
    if non_o_labels:
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_strs, pred_strs, 
            labels=non_o_labels, 
            average='micro',
            zero_division=0
        )
    else:
        precision = recall = f1 = 0.0
    
    accuracy = accuracy_score(label_strs, pred_strs)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def debug_label_distribution(dataset, name="Dataset"):
    """Debug helper to show label distribution BEFORE training"""
    logger.info(f"\n🔍 {name} Label Distribution Check:")
    total_tokens = 0
    label_counts = {}
    
    for i in range(min(5, len(dataset))):
        try:
            item = dataset[i]
            labels = item["labels"].numpy()
            valid_labels = labels[labels != -100]
            
            logger.info(f"  Example {i}: {len(valid_labels)} valid tokens")
            total_tokens += len(valid_labels)
            
            for lid in valid_labels:
                label_name = Config.id2label.get(lid, f"ID{lid}")
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
                
        except Exception as e:
            logger.warning(f"  ⚠️ Example {i} failed: {e}")
            continue
    
    if total_tokens == 0:
        logger.error(f"❌ {name} has NO valid tokens with labels != -100!")
        return False
    
    logger.info(f"  Total valid tokens: {total_tokens}")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_tokens * 100)
        logger.info(f"    {label:20s}: {count:4d} ({pct:5.1f}%)")
    
    non_o_count = sum(count for label, count in label_counts.items() if label != "O")
    if non_o_count == 0:
        logger.error(f"❌ {name} has NO entity labels (only 'O')! Training will fail.")
        return False
    
    return True


from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer, LayoutLMv3ImageProcessor

def train_lilt():
    logger.info("="*70)
    logger.info("🚀 STARTING LILT TRAINING (lilt-roberta-en-base)")
    logger.info("="*70)
    logger.info(f"   Model: {Config.model_name}")
    logger.info(f"   Architecture: LayoutLMv3-based (LiLT uses LayoutLMv3)")
    logger.info(f"   Processor: LayoutLMv3Processor (explicit)")
    logger.info(f"   Label alignment: AUTOMATIC via word_labels parameter")
    logger.info("="*70)
    
    # Detect labels
    logger.info("\n🔍 Auto-detecting labels from training data...")
    try:
        Config.label_list = detect_labels_from_dataset(Config.train_dir)
    except Exception as e:
        logger.error(f"❌ Failed to detect labels: {e}")
        sys.exit(1)
    
    if len(Config.label_list) <= 1:
        logger.error("❌ No entity labels found!")
        sys.exit(1)
    
    Config.id2label = Config.get_id2label()
    Config.label2id = Config.get_label2id()
    
    # Create directories
    Path(Config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(Config.logging_dir).mkdir(parents=True, exist_ok=True)
    
    # ✅ FIX: Use LayoutLMv3Processor explicitly
    logger.info("\n📥 Loading LayoutLMv3Processor...")
    try:
        # Load tokenizer and image processor separately
        tokenizer = LayoutLMv3Tokenizer.from_pretrained(
            Config.model_name,
        )
        
        # Note: LiLT uses LayoutLMv3's image processor
        # We use microsoft/layoutlmv3-base as the source for image processor
        image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            "microsoft/layoutlmv3-base",  # Source of image processing config
            apply_ocr=False,
        )
        
        # Create the processor with both components
        processor = LayoutLMv3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        
        logger.info("✅ LayoutLMv3Processor loaded successfully")
        logger.info(f"   Tokenizer class: {tokenizer.__class__.__name__}")
        logger.info(f"   Image processor class: {image_processor.__class__.__name__}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load processor: {e}")
        
        # Fallback: Try using AutoProcessor but check if it's the right type
        logger.warning("⚠️ Trying AutoProcessor as fallback...")
        try:
            processor = AutoProcessor.from_pretrained(
                Config.model_name,
                apply_ocr=False,
            )
            logger.info(f"   AutoProcessor loaded: {processor.__class__.__name__}")
            
            # Check if it has the required method
            if not hasattr(processor, 'image_processor'):
                logger.error("❌ Loaded processor doesn't have image_processor attribute!")
                raise ValueError("Wrong processor type loaded")
                
        except Exception as e2:
            logger.error(f"❌ Fallback also failed: {e2}")
            logger.error("\n💡 TROUBLESHOOTING:")
            logger.error("   1. Install latest transformers: pip install --upgrade transformers")
            logger.error("   2. Clear cache: rm -rf ~/.cache/huggingface/hub/")
            logger.error("   3. Try specific version: pip install transformers==4.40.0")
            logger.error("   4. Check model compatibility at: https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base")
            sys.exit(1)
    
    # ✅ ADD THIS SECTION: Load datasets
    logger.info("\n📚 Loading datasets...")
    try:
        train_dataset = LILTDataset(Config.train_dir, processor, Config.label2id)
        val_dataset = LILTDataset(Config.val_dir, processor, Config.label2id)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("No valid examples found!")
            
        logger.info(f"✅ Train: {len(train_dataset)} examples | Val: {len(val_dataset)} examples")
        
        # Validate label distribution
        if not debug_label_distribution(train_dataset, "TRAIN"):
            logger.error("❌ Training dataset invalid. Aborting.")
            sys.exit(1)
        if not debug_label_distribution(val_dataset, "VALIDATION"):
            logger.warning("⚠️ Validation set has no entity labels - metrics will be low")
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ✅ ADD THIS SECTION: Load model
    logger.info("\n🧠 Loading LiLT model...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            Config.model_name,
            num_labels=len(Config.label_list),
            id2label=Config.id2label,
            label2id=Config.label2id,
            ignore_mismatched_sizes=True,
        )
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model loaded ({param_count:,} parameters)")
        logger.info(f"   Num labels: {len(Config.label_list)}")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        sys.exit(1)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=Config.logging_dir,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        
        learning_rate=Config.learning_rate,
        num_train_epochs=Config.num_train_epochs,
        warmup_ratio=Config.warmup_ratio,
        weight_decay=Config.weight_decay,
        max_grad_norm=Config.max_grad_norm,
        
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,  # Now this variable is defined!
        args=training_args,
        train_dataset=train_dataset,  # Now this variable is defined!
        eval_dataset=val_dataset,     # Now this variable is defined!
        processing_class=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("\n" + "="*70)
    logger.info("🔥 STARTING TRAINING")
    logger.info("="*70)
    
    try:
        train_result = trainer.train()
        logger.info("\n✅ Training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save
    logger.info("\n💾 Saving final model and processor...")
    trainer.save_model(Config.output_dir)
    processor.save_pretrained(Config.output_dir)
    
    with open(Path(Config.output_dir) / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(Config.id2label, f, indent=2, ensure_ascii=False)
    
    with open(Path(Config.output_dir) / "training_config.json", "w") as f:
        json.dump({
            "model_name": Config.model_name,
            "max_seq_length": Config.max_seq_length,
            "label_list": Config.label_list,
            "num_train_epochs": Config.num_train_epochs,
            "learning_rate": Config.learning_rate,
        }, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"   Model saved to: {Config.output_dir}")
    logger.info(f"   Labels: {Config.label_list}")
    logger.info(f"   Best F1: {trainer.state.best_metric:.4f}" if hasattr(trainer.state, 'best_metric') else "   Best F1: N/A")
    logger.info("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LiLT model for invoice extraction")
    parser.add_argument("--train_dir", type=str, default=Config.train_dir, help="Training data directory")
    parser.add_argument("--val_dir", type=str, default=Config.val_dir, help="Validation data directory")
    parser.add_argument("--output_dir", type=str, default=Config.output_dir, help="Output directory")
    parser.add_argument("--model_name", type=str, default=Config.model_name, help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    Config.train_dir = args.train_dir
    Config.val_dir = args.val_dir
    Config.output_dir = args.output_dir
    Config.model_name = args.model_name
    Config.max_seq_length = args.max_seq_length
    Config.num_train_epochs = args.epochs
    Config.learning_rate = args.learning_rate
    Config.per_device_train_batch_size = args.batch_size
    Config.gradient_accumulation_steps = args.gradient_accumulation_steps
    Config.debug_mode = args.debug
    
    for d in [Config.train_dir, Config.val_dir]:
        if not Path(d).exists():
            logger.error(f"❌ Directory not found: {d}")
            sys.exit(1)
    
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        logger.warning("⚠️ CUDA not available - training will be slow on CPU")
    
    train_lilt()