#!/usr/bin/env python3
"""
lilt_invoice_train_final.py — COMPLETE FIXED VERSION
✅ Token class weights for balanced HEADER/QUESTION/ANSWER learning
✅ Relation class weights to handle no_relation dominance
✅ Proper hidden state extraction for relation extraction
✅ Saves label maps in format expected by lilt_inference_relation_v6.py
✅ Uses eval_f1 for best model selection (HEADER may not be in val set)
✅ Fixed numpy array handling in compute_token_metrics
✅ **CORRECTED BIO TAGGING** – now uses B- for first token, I- for subsequent tokens
"""
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    LayoutLMv3Tokenizer,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import logging

warnings.filterwarnings("ignore", message=".*unexpected key.*")
warnings.filterwarnings("ignore", message=".*missing keys.*")
warnings.filterwarnings("ignore", message=".*Some weights.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class TrainingConfig:
    train_dir: str = "lilt_dataset/train"
    val_dir: str = "lilt_dataset/val"
    images_dir: str = "new_form"
    output_dir: str = "lilt_model_relations"
    model_name: str = "SCUT-DLVCLab/lilt-roberta-en-base"
    max_length: int = 512
    token_labels: List[str] = field(default_factory=lambda: [
        "O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"
    ])
    relation_types: List[str] = field(default_factory=lambda: [
        "no_relation", "relation_right", "relation_left", 
        "relation_above", "relation_below", "relation_unknown"
    ])
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 8
    use_relation_extraction: bool = True
    relation_loss_weight: float = 0.3
    max_entity_pairs: int = 64
    seed: int = 42
    use_cuda: bool = True
    use_token_class_weights: bool = True
    use_relation_class_weights: bool = True
    use_refined_labels: bool = False
    eval_steps: int = 50
    save_total_limit: int = 3


# ============================================================================
# Joint Model: Token Classification + Relation Extraction
# ============================================================================
class LiLTForTokenAndRelationClassification(nn.Module):
    """Joint model compatible with LiLT checkpoint using AutoModel"""
    
    def __init__(self, base_model_name: str, num_token_labels: int, 
                 num_relation_types: int, hidden_size: int = 768, 
                 relation_dropout: float = 0.1,
                 token_class_weights: Optional[torch.Tensor] = None,
                 relation_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        config = AutoConfig.from_pretrained(base_model_name, ignore_mismatched_sizes=True)
        config.num_labels = num_token_labels
        
        self.base = AutoModelForTokenClassification.from_pretrained(
            base_model_name, config=config, ignore_mismatched_sizes=True, torch_dtype=torch.float32)
        self.config = self.base.config
        
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(relation_dropout),
            nn.Linear(hidden_size, num_relation_types)
        )
        
        self.num_token_labels = num_token_labels
        self.num_relation_types = num_relation_types
        self.hidden_size = hidden_size
        
        self.register_buffer('token_class_weights', 
                           token_class_weights if token_class_weights is not None else torch.ones(num_token_labels))
        self.register_buffer('relation_class_weights', 
                           relation_class_weights if relation_class_weights is not None else torch.ones(num_relation_types))
        self.relation_loss_weight = 0.3
        
    def forward(self, input_ids=None, bbox=None, attention_mask=None, 
                pixel_values=None, labels=None, entity_pairs=None, 
                relation_labels=None, **kwargs):
        
        outputs = self.base(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, 
            pixel_values=pixel_values, labels=None, output_hidden_states=True, return_dict=True)
        
        token_logits = outputs.logits
        result = {'logits': token_logits}
        
        if labels is not None:
            logits_len = token_logits.size(1)
            labels_len = labels.size(1) if labels.dim() == 2 else labels.size(0)
            if logits_len != labels_len:
                if labels_len < logits_len:
                    pad_len = logits_len - labels_len
                    if labels.dim() == 1:
                        padding = torch.full((pad_len,), -100, dtype=labels.dtype, device=labels.device)
                        labels = torch.cat([labels, padding], dim=0)
                    else:
                        padding = torch.full((labels.size(0), pad_len), -100, dtype=labels.dtype, device=labels.device)
                        labels = torch.cat([labels, padding], dim=1)
                else:
                    labels = labels[:, :logits_len] if labels.dim() > 1 else labels[:logits_len]
            loss_fct = nn.CrossEntropyLoss(weight=self.token_class_weights, ignore_index=-100)
            token_logits_flat = token_logits.view(-1, self.num_token_labels)
            labels_flat = labels.view(-1)
            token_loss = loss_fct(token_logits_flat, labels_flat)
            result['token_loss'] = token_loss
            result['loss'] = token_loss
        
        if entity_pairs is not None and self.training:
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states and len(outputs.hidden_states) > 0:
                sequence_output = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                sequence_output = outputs.last_hidden_state
            else:
                backbone_kwargs = {'input_ids': input_ids, 'bbox': bbox, 'attention_mask': attention_mask, 'output_hidden_states': True, 'return_dict': True}
                if pixel_values is not None:
                    backbone_kwargs['pixel_values'] = pixel_values
                try:
                    backbone_out = self.base.base_model(**backbone_kwargs)
                    sequence_output = backbone_out.last_hidden_state if hasattr(backbone_out, 'last_hidden_state') else backbone_out[0]
                except (TypeError, AttributeError):
                    backbone_kwargs.pop('pixel_values', None)
                    backbone_out = self.base.base_model(**backbone_kwargs)
                    sequence_output = backbone_out.last_hidden_state if hasattr(backbone_out, 'last_hidden_state') else backbone_out[0]
            
            rel_logits, rel_loss = self._compute_relation_loss(sequence_output, entity_pairs, relation_labels, attention_mask)
            if rel_loss is not None:
                result['relation_loss'] = rel_loss
                result['loss'] = result.get('loss', 0) + self.relation_loss_weight * rel_loss
        
        return result
    
    def _compute_relation_loss(self, sequence_output: torch.Tensor, entity_pairs: List[List[Tuple[int, int]]], relation_labels: Optional[torch.Tensor], attention_mask: torch.Tensor):
        batch_size = sequence_output.size(0)
        max_pairs = max((len(p) for p in entity_pairs), default=0)
        if max_pairs == 0:
            return None, None
        
        cls_repr = sequence_output[:, 0, :]
        relation_reprs, valid_mask = [], []
        
        for b in range(batch_size):
            pairs = entity_pairs[b] if b < len(entity_pairs) else []
            batch_reprs, batch_valid = [], []
            for i in range(max_pairs):
                if i < len(pairs) and pairs[i][0] < sequence_output.size(1):
                    head_idx, tail_idx = pairs[i]
                    head_idx = min(head_idx, sequence_output.size(1) - 1)
                    tail_idx = min(tail_idx, sequence_output.size(1) - 1)
                    head_repr = sequence_output[b, head_idx, :]
                    tail_repr = sequence_output[b, tail_idx, :]
                    pair_repr = torch.cat([cls_repr[b], head_repr, tail_repr], dim=-1)
                    batch_reprs.append(pair_repr)
                    batch_valid.append(1)
                else:
                    batch_reprs.append(torch.zeros(self.hidden_size * 3, device=sequence_output.device))
                    batch_valid.append(0)
            relation_reprs.append(torch.stack(batch_reprs))
            valid_mask.append(torch.tensor(batch_valid, device=sequence_output.device))
        
        relation_reprs = torch.stack(relation_reprs)
        valid_mask = torch.stack(valid_mask)
        relation_logits = self.relation_classifier(relation_reprs)
        
        relation_loss = None
        if relation_labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.relation_class_weights, reduction='none')
            flat_logits = relation_logits.view(-1, self.num_relation_types)
            flat_labels = relation_labels.view(-1)
            flat_valid = valid_mask.view(-1)
            per_pair_loss = loss_fct(flat_logits, flat_labels)
            relation_loss = (per_pair_loss * flat_valid).sum() / (flat_valid.sum() + 1e-8)
        
        return relation_logits, relation_loss


# ============================================================================
# Dataset — FIXED BIO TAGGING
# ============================================================================
class LiLTInvoiceRelationDataset(Dataset):
    def __init__(self, data_dir: str, images_dir: str, processor: LayoutLMv3Processor, token_labels: List[str], relation_types: List[str], max_length: int = 512, max_entity_pairs: int = 64, use_refined_labels: bool = False):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.token_labels = token_labels
        self.relation_types = relation_types
        self.max_length = max_length
        self.max_entity_pairs = max_entity_pairs
        self.use_refined_labels = use_refined_labels
        self.token_label2id = {l.strip(): i for i, l in enumerate(token_labels)}
        self.relation_type2id = {t.strip(): i for i, t in enumerate(relation_types)}
        self.id2token_label = {i: l for l, i in self.token_label2id.items()}
        self.examples = self._load_data()
        logger.info(f"✅ Loaded {len(self.examples)} examples from {data_dir}")
    
    def _load_data(self) -> List[Dict]:
        examples = []
        json_files = [f for f in self.data_dir.glob("*.json") if f.name not in ['index.json', 'metadata.json']]
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tasks = data if isinstance(data, list) else [data]
                for task in tasks:
                    example = self._parse_task(task)
                    if example:
                        examples.append(example)
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse {json_file.name}: {e}")
        return examples
    
    def _parse_task(self, task: Dict) -> Optional[Dict]:
        try:
            task = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in task.items()}
            image_path = None
            for key in ['image_path', 'image']:
                if key in task and task[key]:
                    rel_path = str(task[key]).strip()
                    image_name = Path(rel_path).name
                    if '-' in image_name and len(image_name.split('-')[0]) <= 10:
                        image_name = image_name.split('-', 1)[-1]
                    candidate = self.images_dir / image_name
                    if candidate.exists():
                        image_path = candidate
                        break
            if not image_path or not image_path.exists():
                return None
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size
            tokens = task.get('tokens', task.get('tokens ', []))
            if not tokens:
                return None
            
            # Ensure tokens are sorted in reading order (top-to-bottom, left-to-right)
            tokens.sort(key=lambda t: (t.get('center_y', t['box'][1]), t.get('center_x', t['box'][0])))
            
            id_to_sorted_idx = {}
            for idx, token in enumerate(tokens):
                token = {k.strip(): v for k, v in token.items()}
                token_id = token.get('id')
                if token_id is not None:
                    id_to_sorted_idx[token_id] = idx
            
            words, boxes, token_labels = [], [], []
            prev_label = None  # track previous token's base label for BIO
            
            for token in tokens:
                token = {k.strip(): v for k, v in token.items()}
                # Choose label source
                label = str(token.get('refined_label' if self.use_refined_labels and 'refined_label' in token else 'label', 'O')).strip().upper()
                box = token.get('box', [0, 0, 0, 0])
                text = str(token.get('text', '')).strip()
                
                norm_box = [
                    max(0, min(1000, int(1000 * box[0] / img_w))),
                    max(0, min(1000, int(1000 * box[1] / img_h))),
                    max(0, min(1000, int(1000 * box[2] / img_w))),
                    max(0, min(1000, int(1000 * box[3] / img_h)))
                ]
                
                if not text or text.isspace():
                    text = '[TOKEN]'
                words.append(text)
                boxes.append(norm_box)
                
                # ----- CORRECT BIO ASSIGNMENT -----
                if label == 'O':
                    bio = 'O'
                    prev_label = None
                else:
                    # label is one of HEADER, QUESTION, ANSWER
                    if prev_label == label:
                        # same entity continues → I- tag
                        bio = f"I-{label}"
                    else:
                        # new entity starts → B- tag
                        bio = f"B-{label}"
                    prev_label = label
                
                label_id = self.token_label2id.get(bio, self.token_label2id['O'])
                token_labels.append(label_id)
            
            if not words:
                return None
            
            relations = task.get('relations', task.get('relations ', []))
            entity_pairs, relation_label_ids = [], []
            for rel in relations:
                rel = {k.strip(): v for k, v in rel.items()}
                head_orig, tail_orig = rel.get('head'), rel.get('tail')
                rel_type = str(rel.get('type', 'relation')).strip()
                direction = str(rel.get('direction', '')).strip()
                if head_orig not in id_to_sorted_idx or tail_orig not in id_to_sorted_idx:
                    continue
                head_idx, tail_idx = id_to_sorted_idx[head_orig], id_to_sorted_idx[tail_orig]
                if head_idx == tail_idx:
                    continue
                rel_key = f"{rel_type}_{direction}".lower().replace(' ', '_') if direction else rel_type.lower()
                rel_id = self.relation_type2id.get(rel_key, self.relation_type2id.get('relation_right', 1))
                entity_pairs.append((head_idx, tail_idx))
                relation_label_ids.append(rel_id)
            
            # Pad to max_entity_pairs
            while len(entity_pairs) < self.max_entity_pairs:
                entity_pairs.append((0, 0))
                relation_label_ids.append(self.relation_type2id['no_relation'])
            entity_pairs = entity_pairs[:self.max_entity_pairs]
            relation_label_ids = relation_label_ids[:self.max_entity_pairs]
            
            return {
                'image_path': str(image_path),
                'words': words, 'boxes': boxes, 'token_labels': token_labels,
                'entity_pairs': entity_pairs, 'relation_labels': relation_label_ids,
                'task_id': task.get('id', 'unknown'),
            }
        except Exception as e:
            logger.warning(f"⚠️ Parse error: {e}", exc_info=True)
            return None
    
    def __len__(self): return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        try:
            image = Image.open(example['image_path']).convert("RGB")
            encoding = self.processor(images=image, text=example['words'], boxes=example['boxes'], word_labels=example['token_labels'], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            result = {}
            for k, v in encoding.items():
                if isinstance(v, torch.Tensor) and v.size(0) == 1:
                    result[k] = v.squeeze(0)
                else:
                    result[k] = v
            result['entity_pairs'] = torch.tensor(example['entity_pairs'], dtype=torch.long)
            result['relation_labels'] = torch.tensor(example['relation_labels'], dtype=torch.long)
            result['task_id'] = example['task_id']
            return result
        except Exception as e:
            logger.error(f"❌ Getitem error: {e}", exc_info=True)
            return self._dummy()
    
    def _dummy(self) -> Dict:
        return {'pixel_values': torch.zeros(3, 224, 224), 'input_ids': torch.zeros(self.max_length, dtype=torch.long), 'attention_mask': torch.zeros(self.max_length, dtype=torch.long), 'bbox': torch.zeros(self.max_length, 4, dtype=torch.long), 'labels': torch.full((self.max_length,), -100, dtype=torch.long), 'entity_pairs': torch.zeros((self.max_entity_pairs, 2), dtype=torch.long), 'relation_labels': torch.zeros(self.max_entity_pairs, dtype=torch.long), 'task_id': 'dummy'}


# ============================================================================
# Data Collator
# ============================================================================
class RelationDataCollator:
    def __init__(self, processor: LayoutLMv3Processor, max_entity_pairs: int = 64):
        self.processor = processor
        self.max_entity_pairs = max_entity_pairs
    def __call__(self, features: List[Dict]) -> Dict:
        batch = {}
        relation_fields = ['entity_pairs', 'relation_labels', 'task_id']
        for key in features[0].keys():
            if key in relation_fields:
                batch[key] = [f[key] for f in features]
            else:
                batch[key] = torch.stack([f[key] for f in features], dim=0)
        batch['entity_pairs'] = torch.stack(batch['entity_pairs'], dim=0)
        batch['relation_labels'] = torch.stack(batch['relation_labels'], dim=0)
        return batch


# ============================================================================
# Metrics - FIXED for all numpy array issues
# ============================================================================
def compute_token_metrics(eval_pred, id2label: Dict[int, str]):
    if hasattr(eval_pred, 'predictions'):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=2)
    true_labels, true_preds = [], []
    if isinstance(labels, (list, tuple)):
        for lbl_seq, pred_seq in zip(labels, predictions):
            lbl_arr = np.asarray(lbl_seq) if not isinstance(lbl_seq, np.ndarray) else lbl_seq
            pred_arr = np.asarray(pred_seq) if not isinstance(pred_seq, np.ndarray) else pred_seq
            for lbl, pred in zip(lbl_arr.ravel(), pred_arr.ravel()):
                if lbl != -100:
                    true_labels.append(int(lbl))
                    true_preds.append(int(pred))
    else:
        labels = np.asarray(labels)
        labels_flat = labels.ravel()
        predictions_flat = predictions.ravel()
        mask = labels_flat != -100
        true_labels = labels_flat[mask].tolist()
        true_preds = predictions_flat[mask].tolist()
    if not true_labels:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(true_labels, true_preds)
    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
    unique_labels = sorted(set(true_labels))
    present_target_names = [id2label.get(i, str(i)) for i in unique_labels]
    try:
        report = classification_report(true_labels, true_preds, labels=unique_labels, target_names=present_target_names, output_dict=True, zero_division=0)
        for label_name in ['B-HEADER', 'B-QUESTION', 'B-ANSWER', 'O']:
            if label_name in report:
                metrics[f'{label_name}_f1'] = report[label_name]['f1-score']
    except ValueError as e:
        logger.warning(f"⚠️ classification_report failed: {e}, using basic metrics only")
    return metrics


def compute_token_class_weights(dataset: LiLTInvoiceRelationDataset) -> torch.Tensor:
    counter = Counter()
    for ex in dataset.examples:
        for lbl_id in ex['token_labels']:
            counter[lbl_id] += 1
    num_classes = len(dataset.token_labels)
    for i in range(num_classes):
        if i not in counter:
            counter[i] = 1
    total = sum(counter.values())
    weights = torch.tensor([total / (counter[i] * num_classes) for i in range(num_classes)], dtype=torch.float)
    weights = weights / weights.mean()
    label_names = dataset.token_labels
    weight_info = {label_names[i]: round(float(weights[i]), 3) for i in range(num_classes)}
    logger.info(f"⚖️ Token class weights: {weight_info}")
    return weights


def compute_relation_class_weights(dataset: LiLTInvoiceRelationDataset) -> torch.Tensor:
    counter = Counter()
    for ex in dataset.examples:
        for rel_id in ex['relation_labels']:
            counter[rel_id] += 1
    num_classes = len(dataset.relation_types)
    for i in range(num_classes):
        if i not in counter:
            counter[i] = 1
    total = sum(counter.values())
    weights = torch.tensor([total / (counter[i] * num_classes) for i in range(num_classes)], dtype=torch.float)
    weights = weights / weights.mean()
    label_names = dataset.relation_types
    weight_info = {label_names[i]: round(float(weights[i]), 3) for i in range(num_classes)}
    logger.info(f"⚖️ Relation class weights: {weight_info}")
    return weights


# ============================================================================
# Custom Trainer
# ============================================================================
class JointTrainer(Trainer):
    def __init__(self, *args, config: TrainingConfig = None, id2token_label: Dict = None, **kwargs):
        kwargs.pop('tokenizer', None)
        super().__init__(*args, **kwargs)
        self.config = config
        self.id2token_label = id2token_label or {}
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_training = self.model.training
        entity_pairs = inputs.pop('entity_pairs', None)
        relation_labels = inputs.pop('relation_labels', None)
        model.relation_loss_weight = self.config.relation_loss_weight
        outputs = model(input_ids=inputs.get('input_ids'), bbox=inputs.get('bbox'), attention_mask=inputs.get('attention_mask'), pixel_values=inputs.get('pixel_values'), labels=inputs.get('labels'), entity_pairs=entity_pairs if is_training else None, relation_labels=relation_labels if is_training else None)
        loss = outputs.get('loss')
        if return_outputs:
            return loss, {'logits': outputs.get('logits')}
        return loss


# ============================================================================
# Training Function
# ============================================================================
def train(config: TrainingConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Device: {device} | Relations: {config.use_relation_extraction}")
    for d in [config.train_dir, config.val_dir, config.images_dir]:
        if not Path(d).exists():
            logger.error(f"❌ Missing: {d}")
            return None
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"\n📥 Loading processor...")
    tokenizer = LayoutLMv3Tokenizer.from_pretrained(config.model_name, add_prefix_space=True)
    image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    processor = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
    logger.info(f"\n📚 Loading datasets...")
    train_dataset = LiLTInvoiceRelationDataset(config.train_dir, config.images_dir, processor, config.token_labels, config.relation_types, config.max_length, config.max_entity_pairs, use_refined_labels=config.use_refined_labels)
    val_dataset = LiLTInvoiceRelationDataset(config.val_dir, config.images_dir, processor, config.token_labels, config.relation_types, config.max_length, config.max_entity_pairs, use_refined_labels=config.use_refined_labels)
    if len(train_dataset) == 0:
        logger.error("❌ No training examples!")
        return None
    logger.info(f"✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    token_class_weights = None
    if config.use_token_class_weights:
        token_class_weights = compute_token_class_weights(train_dataset).to(device)
    relation_class_weights = None
    if config.use_relation_extraction and config.use_relation_class_weights:
        relation_class_weights = compute_relation_class_weights(train_dataset).to(device)
    logger.info(f"\n🤖 Loading joint model...")
    model = LiLTForTokenAndRelationClassification(base_model_name=config.model_name, num_token_labels=len(config.token_labels), num_relation_types=len(config.relation_types), token_class_weights=token_class_weights, relation_class_weights=relation_class_weights)
    logger.info(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    total_steps = len(train_dataset) * config.num_train_epochs // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    warmup_steps = int(total_steps * config.warmup_ratio)
    training_args = TrainingArguments(output_dir=config.output_dir, num_train_epochs=config.num_train_epochs, per_device_train_batch_size=config.per_device_train_batch_size, per_device_eval_batch_size=config.per_device_eval_batch_size, learning_rate=config.learning_rate, weight_decay=config.weight_decay, warmup_steps=warmup_steps, lr_scheduler_type='linear', gradient_accumulation_steps=config.gradient_accumulation_steps, eval_strategy="steps", eval_steps=config.eval_steps, save_strategy="steps", save_steps=config.eval_steps, load_best_model_at_end=True, metric_for_best_model="eval_f1", greater_is_better=True, logging_steps=10, save_total_limit=config.save_total_limit, fp16=torch.cuda.is_available(), dataloader_num_workers=0, report_to="none", seed=config.seed, remove_unused_columns=False, prediction_loss_only=False)
    data_collator = RelationDataCollator(processor, config.max_entity_pairs)
    trainer_kwargs = dict(model=model, args=training_args, config=config, id2token_label=train_dataset.id2token_label, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator, compute_metrics=lambda x: compute_token_metrics(x, train_dataset.id2token_label))
    try:
        trainer_kwargs['processing_class'] = processor
    except TypeError:
        pass
    trainer = JointTrainer(**trainer_kwargs)
    logger.info(f"\n{'='*60}\n🔥 STARTING TRAINING\n{'='*60}")
    train_result = trainer.train()
    logger.info(f"\n💾 Saving model and label maps...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    label_maps = {'label_map': {str(i): l for i, l in enumerate(config.token_labels)}, 'relation_label_map': {str(i): t for i, t in enumerate(config.relation_types)}, 'config': {'token_labels': config.token_labels, 'relation_types': config.relation_types, 'model_name': config.model_name}}
    with open(Path(config.output_dir) / "label_map.json", 'w') as f:
        json.dump(label_maps['label_map'], f, indent=2)
    with open(Path(config.output_dir) / "relation_label_map.json", 'w') as f:
        json.dump(label_maps['relation_label_map'], f, indent=2)
    with open(Path(config.output_dir) / "label_maps.json", 'w') as f:
        json.dump(label_maps, f, indent=2)
    logger.info(f"\n📊 Final Training Results:")
    for k, v in train_result.metrics.items():
        if isinstance(v, float): logger.info(f"   {k}: {v:.4f}")
    logger.info(f"\n🔍 Running final evaluation...")
    eval_result = trainer.evaluate()
    for k, v in eval_result.items():
        if isinstance(v, float): logger.info(f"   {k}: {v:.4f}")
    logger.info(f"\n✅ Done! Model saved to: {config.output_dir}")
    return trainer


# ============================================================================
# CLI
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="LiLT Invoice Training — Fixed")
    parser.add_argument("--train_dir", default="lilt_dataset10/train")
    parser.add_argument("--val_dir", default="lilt_dataset10/val")
    parser.add_argument("--images_dir", default="new_form")
    parser.add_argument("--output_dir", default="lilt_model_10")
    parser.add_argument("--model_name", default="SCUT-DLVCLab/lilt-roberta-en-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--relation_loss_weight", type=float, default=0.3)
    parser.add_argument("--max_entity_pairs", type=int, default=64)
    parser.add_argument("--no_relations", action="store_true")
    parser.add_argument("--no_token_class_weights", action="store_true")
    parser.add_argument("--no_relation_class_weights", action="store_true")
    parser.add_argument("--use_refined_labels", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(train_dir=args.train_dir, val_dir=args.val_dir, images_dir=args.images_dir, output_dir=args.output_dir, model_name=args.model_name, max_length=args.max_length, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, gradient_accumulation_steps=args.gradient_accumulation_steps, relation_loss_weight=args.relation_loss_weight, max_entity_pairs=args.max_entity_pairs, use_relation_extraction=not args.no_relations, seed=args.seed, use_cuda=not args.no_cuda, use_token_class_weights=not args.no_token_class_weights, use_relation_class_weights=not args.no_relation_class_weights, use_refined_labels=args.use_refined_labels, eval_steps=args.eval_steps)
    print("="*70)
    print(f"🧾 LiLT Invoice Training — Fixed")
    print("="*70)
    print(f"📁 Train: {config.train_dir} | Val: {config.val_dir}")
    print(f"🖼️  Images: {config.images_dir}")
    print(f"🤖 Model: {config.model_name}")
    print(f"⏱️  Epochs: {config.num_train_epochs} | LR: {config.learning_rate}")
    print(f"🔗 Relations: {config.use_relation_extraction} (weight: {config.relation_loss_weight})")
    print(f"⚖️  Token class weights: {config.use_token_class_weights}")
    print(f"⚖️  Relation class weights: {config.use_relation_class_weights}")
    print(f"🔖 Use refined labels: {config.use_refined_labels}")
    print(f"📊 Eval every {config.eval_steps} steps")
    print("="*70)
    trainer = train(config)
    if trainer:
        print(f"\n✅ Training complete! Test with:")
        print(f"   python lilt_inference_relation_v6.py your_invoice.png --model {config.output_dir}")


if __name__ == "__main__":
    main()