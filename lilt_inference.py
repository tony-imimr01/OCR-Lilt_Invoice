#!/usr/bin/env python3
"""
lilt_invoice_complete_fixed_v5.py — Fixed address and financial extraction
FIXES:
✅ Fixed vendor extraction: Filters out address labels
✅ Fixed address mixing: Better boundary detection
✅ Fixed shipping amount: Stricter label verification
✅ Added Notes field extraction
✅ Enhanced Order ID detection
"""

import json
import sys
from pathlib import Path
import re
from datetime import datetime
from PIL import Image, ImageDraw
import torch
import easyocr
from typing import Dict, List, Optional, Tuple
import gc

import PIL.Image as PIL_Image

from transformers import AutoTokenizer, AutoModelForTokenClassification

class LiLTInvoiceParser:
    def __init__(self, model_path: str, use_cuda: bool = True):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # ✅ Load label map FIRST
        self.id2label = self._load_label_map()
        print(f"📋 Model labels: {set(self.id2label.values())}")
        
        # ✅ FIX: Use LayoutLMv3Processor OR DonutProcessor, not AutoProcessor
        # Most LiLT models use LayoutLMv3Processor
        try:
            from transformers import LayoutLMv3Processor
            self.processor = LayoutLMv3Processor.from_pretrained(
                model_path,
                apply_ocr=False,  # CRITICAL: We provide our own OCR
                use_fast=True,
            )
            print(f"✅ Using LayoutLMv3Processor")
        except Exception as e:
            print(f"⚠️ LayoutLMv3Processor failed: {e}, trying LayoutLMv2Processor")
            try:
                from transformers import LayoutLMv2Processor
                self.processor = LayoutLMv2Processor.from_pretrained(
                    model_path,
                    apply_ocr=False,
                    revision="no_ocr",
                )
                print(f"✅ Using LayoutLMv2Processor")
            except Exception as e2:
                print(f"⚠️ LayoutLMv2Processor failed: {e2}, falling back to tokenizer")
                self.processor = AutoTokenizer.from_pretrained(
                    model_path,
                    add_prefix_space=True,
                )
        
        # ✅ Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16 if use_cuda and torch.cuda.is_available() else torch.float32,
        )
        
        # ✅ Device setup
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # ✅ OCR setup
        try:
            if use_cuda and torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 4 * 1024**3:
                self.reader = easyocr.Reader(['en'], gpu=True)
            else:
                self.reader = easyocr.Reader(['en'], gpu=False)
                print("📝 EasyOCR running on CPU to save GPU memory")
        except Exception as e:
            print(f"⚠️ EasyOCR init failed: {e}")
            self.reader = None
        
        print(f"✅ Model loaded | Device: {self.device} | Processor: {self.processor.__class__.__name__}")

    def _get_model_size_mb(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2
    
    def _load_label_map(self) -> dict:
        label_map_path = self.model_path / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                return {int(k): v for k, v in json.load(f).items()}
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_path)
        if hasattr(config, 'id2label') and config.id2label:
            return config.id2label
        
        return {
            0: "O",
            1: "B-HEADER", 2: "I-HEADER",
            3: "B-QUESTION", 4: "I-QUESTION",
            5: "B-ANSWER", 6: "I-ANSWER",
            7: "B-OTHER", 8: "I-OTHER"
        }
    
    def extract_ocr_tokens(self, image_path: str) -> Tuple[List[Dict], Tuple[int, int]]:
        """Extract tokens with numeric/currency detection"""
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        tokens = []
        
        if self.reader:
            try:
                results = self.reader.readtext(
                    str(image_path),
                    detail=1,
                    paragraph=False,
                    min_size=8
                )
                
                for bbox, text, conf in results:
                    if not text.strip() or conf < 0.25:
                        continue
                    
                    x0 = int(min(p[0] for p in bbox))
                    y0 = int(min(p[1] for p in bbox))
                    x1 = int(max(p[0] for p in bbox))
                    y1 = int(max(p[1] for p in bbox))
                    
                    if (x1 - x0) < 3 or (y1 - y0) < 3:
                        continue
                    
                    norm_bbox = [
                        max(0, min(1000, int(1000 * x0 / w))),
                        max(0, min(1000, int(1000 * y0 / h))),
                        max(0, min(1000, int(1000 * x1 / w))),
                        max(0, min(1000, int(1000 * y1 / h))),
                    ]
                    
                    tokens.append({
                        "text": text.strip(),
                        "bbox": [x0, y0, x1, y1],
                        "norm_bbox": norm_bbox,
                        "center_x": (x0 + x1) / 2,
                        "center_y": (y0 + y1) / 2,
                        "confidence": conf,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "is_numeric": self._is_numeric_token(text),
                        "is_currency": self._contains_currency(text),
                    })
                
                tokens.sort(key=lambda t: (t["center_y"], t["center_x"]))
                return tokens, (w, h)
                
            except Exception as e:
                print(f"⚠️ EasyOCR failed: {e}")
        
        # Fallback to pytesseract
        try:
            import pytesseract
            from pytesseract import Output
            
            ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT, config='--psm 6')
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if not text or int(ocr_data['conf'][i]) < 50:
                    continue
                
                x, y = ocr_data['left'][i], ocr_data['top'][i]
                w_t, h_t = ocr_data['width'][i], ocr_data['height'][i]
                
                if w_t < 3 or h_t < 3:
                    continue
                
                tokens.append({
                    "text": text,
                    "bbox": [x, y, x + w_t, y + h_t],
                    "norm_bbox": [
                        max(0, min(1000, int(1000 * x / w))),
                        max(0, min(1000, int(1000 * y / h))),
                        max(0, min(1000, int(1000 * (x + w_t) / w))),
                        max(0, min(1000, int(1000 * (y + h_t) / h))),
                    ],
                    "center_x": x + w_t / 2,
                    "center_y": y + h_t / 2,
                    "confidence": int(ocr_data['conf'][i]) / 100,
                    "width": w_t,
                    "height": h_t,
                    "is_numeric": self._is_numeric_token(text),
                    "is_currency": self._contains_currency(text),
                })
            
            tokens.sort(key=lambda t: (t["center_y"], t["center_x"]))
            return tokens, (w, h)
            
        except Exception as e:
            print(f"⚠️ PyTesseract failed: {e}")
            return [], (w, h)
    
    def _is_numeric_token(self, text: str) -> bool:
        clean = re.sub(r'[^\d.,]', '', text)
        return bool(re.match(r'^\d+\.?\d*$', clean)) and len(clean) >= 1
    
    def _contains_currency(self, text: str) -> bool:
        return bool(re.search(r'[\$\€\£\¥\₹]|USD|EUR|GBP|JPY|INR', text))
    
    def _merge_adjacent_currency_tokens(self, tokens: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Merge split currency tokens ($ + 1,234.56)"""
        merged = []
        i = 0
        horiz_tol = img_w * 0.04
        vert_tol = img_h * 0.06
        
        while i < len(tokens):
            curr = tokens[i]
            if curr["is_currency"] and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if (abs(nxt["center_y"] - curr["center_y"]) < vert_tol and
                    0 < (nxt["center_x"] - curr["center_x"]) < horiz_tol and
                    nxt["is_numeric"]):
                    
                    merged_text = curr["text"] + nxt["text"]
                    merged.append({
                        **curr,
                        "text": merged_text,
                        "bbox": [
                            curr["bbox"][0],
                            min(curr["bbox"][1], nxt["bbox"][1]),
                            nxt["bbox"][2],
                            max(curr["bbox"][3], nxt["bbox"][3]),
                        ],
                        "center_x": (curr["center_x"] + nxt["center_x"]) / 2,
                        "center_y": (curr["center_y"] + nxt["center_y"]) / 2,
                        "width": nxt["bbox"][2] - curr["bbox"][0],
                        "is_numeric": True,
                        "is_currency": True,
                    })
                    i += 2
                    continue
            
            merged.append(curr)
            i += 1
        
        return merged

    def predict(self, image_path: str) -> dict:
        print(f"\n📄 Processing: {Path(image_path).name}")
        
        # 1. Extract OCR tokens
        raw_tokens, (img_w, img_h) = self.extract_ocr_tokens(image_path)
        if not raw_tokens:
            print("⚠️ No text detected")
            return self._empty_result(image_path, img_w, img_h)
        
        tokens = self._merge_adjacent_currency_tokens(raw_tokens, img_w, img_h)
        print(f"   Size: {img_w}x{img_h} | Tokens: {len(tokens)} (merged from {len(raw_tokens)})")
        
        # ✅ DEBUG: Show first few tokens
        print(f"   Sample tokens (first 10):")
        for i, token in enumerate(tokens[:10]):
            print(f"     {i}: '{token['text']}' @ ({token['center_x']:.0f}, {token['center_y']:.0f})")
        
        # 2. Prepare inputs CORRECTLY for the model
        words = [t["text"] for t in tokens]
        boxes = [t["norm_bbox"] for t in tokens]
        
        # ✅ Use PIL_Image (explicit global reference)
        image = PIL_Image.open(image_path).convert("RGB")
        
        # ✅ CRITICAL: Check what type of processor we have
        if hasattr(self.processor, 'image_processor'):
            # It's a LayoutLM/LiLT processor
            print(f"   Using multimodal processor with image...")
            
            # Try different encoding methods
            try:
                # Method 1: Standard LayoutLMv3 approach
                encoding = self.processor(
                    image, 
                    words, 
                    boxes=boxes,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            except Exception as e:
                print(f"   Method 1 failed: {e}, trying Method 2...")
                # Method 2: Try without image first
                try:
                    encoding = self.processor(
                        text=words,
                        boxes=boxes,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    # Manually add pixel_values if needed
                    if not hasattr(encoding, 'pixel_values'):
                        import torchvision.transforms as transforms
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                        ])
                        pixel_values = transform(image).unsqueeze(0)
                        encoding['pixel_values'] = pixel_values
                except Exception as e2:
                    print(f"   Method 2 failed: {e2}, trying Method 3...")
                    # Method 3: Fallback - use tokenizer only
                    encoding = self.processor(
                        text=words,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
        else:
            # It's just a tokenizer
            print(f"   Using tokenizer-only approach...")
            encoding = self.processor(
                text=words,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        
        # Move to device
        encoding = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in encoding.items()}
        
        print(f"   Encoding keys: {encoding.keys()}")
        print(f"   Input IDs shape: {encoding['input_ids'].shape if 'input_ids' in encoding else 'N/A'}")
        
        # 3. Run inference
        with torch.no_grad():
            try:
                outputs = self.model(**encoding)
                
                # Get predictions
                preds = outputs.logits.argmax(-1)
                
                # Handle batch dimension
                if len(preds.shape) == 3:  # [batch, seq_len, num_labels]?
                    preds = preds.squeeze(0)  # Remove batch
                elif len(preds.shape) == 2:  # [batch, seq_len]
                    preds = preds.squeeze(0) if preds.shape[0] == 1 else preds[0]
                
                # Convert to numpy
                preds = preds.cpu().numpy()
                
                # Get probabilities for confidence
                probs = torch.softmax(outputs.logits, dim=-1)
                if len(probs.shape) == 3:
                    probs = probs.squeeze(0)
                probs = probs.cpu().numpy()
                
                # ✅ DEBUG: Print shapes
                print(f"   Predictions shape: {preds.shape}")
                print(f"   Probabilities shape: {probs.shape}")
                if len(probs.shape) == 2:
                    print(f"   Num labels: {probs.shape[1]}")
                
            except Exception as e:
                print(f"❌ Model inference failed: {e}")
                import traceback
                traceback.print_exc()
                return self._empty_result(image_path, img_w, img_h)
        
        # 4. Map predictions to tokens
        predictions = []
        
        # Get word IDs for token alignment
        if 'word_ids' in encoding:
            word_ids = encoding.word_ids(batch_index=0)
        else:
            # Create simple alignment (1:1)
            word_ids = list(range(len(tokens)))
            if len(word_ids) > 512:  # Truncate if too long
                word_ids = word_ids[:512]
        
        print(f"   Word IDs length: {len(word_ids)}, Predictions length: {len(preds)}")
        
        # Align predictions with tokens
        seen = set()
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen or word_idx >= len(tokens):
                continue
            
            seen.add(word_idx)
            
            # Get label
            if idx < len(preds):
                label_id = int(preds[idx])
                label = self.id2label.get(label_id, "O")
                
                # ✅ FIX: Safely extract confidence value
                confidence = 0.0
                if idx < len(probs):
                    try:
                        # Use proper 2D indexing
                        confidence_value = probs[idx, label_id]
                        
                        # Handle different types
                        if isinstance(confidence_value, np.ndarray):
                            if confidence_value.ndim == 0:
                                # 0-d array (scalar)
                                confidence = float(confidence_value)
                            else:
                                # Multi-dimensional array
                                confidence = float(confidence_value.ravel()[0])
                        elif hasattr(confidence_value, 'item'):
                            # NumPy scalar
                            confidence = float(confidence_value.item())
                        else:
                            # Plain Python scalar
                            confidence = float(confidence_value)
                    except Exception as e:
                        print(f"   Warning: Could not extract confidence at idx {idx}: {e}")
                        confidence = 0.5  # Default confidence
                
                predictions.append({
                    **tokens[word_idx],
                    "label": label,
                    "confidence": confidence,
                })
        
        # 5. Diagnostics
        entity_preds = [p for p in predictions if p["label"] != "O"]
        print(f"📊 Entities detected: {len(entity_preds)}")
        
        if entity_preds:
            print(f"   Entity breakdown:")
            for label_type in set([p["label"] for p in entity_preds]):
                count = sum(1 for p in entity_preds if p["label"] == label_type)
                print(f"     {label_type}: {count}")
            # Show first few entities
            for i, pred in enumerate(entity_preds[:5]):
                print(f"     {i}: '{pred['text']}' -> {pred['label']} ({pred['confidence']:.2f})")
        
        self._diagnose_invoice(predictions, img_w, img_h)
        
        # 6. Extract structured data
        structured_data = self._extract_invoice_complete(predictions, img_w, img_h)
        vis_image = self._visualize_predictions(Image.open(image_path).convert("RGB"), predictions)
        
        return {
            "image_path": str(Path(image_path).resolve()),
            "image_size": {"width": img_w, "height": img_h},
            "raw_predictions": predictions,
            "entity_count": len(entity_preds),
            "structured_data": structured_data,
            "visualization": vis_image,
        }

    def _diagnose_invoice(self, predictions: List[Dict], img_w: int, img_h: int):
        """Enhanced diagnostics with safe column detection"""
        print(f"\n🔍 INVOICE STRUCTURE ANALYSIS")
        print("="*70)
        
        numeric_tokens = [p for p in predictions if p.get("is_numeric")]
        if numeric_tokens:
            x_positions = sorted([p["center_x"] for p in numeric_tokens])
            columns = []
            if x_positions:
                current_col = [x_positions[0]]
                for x in x_positions[1:]:
                    if x - current_col[-1] < img_w * 0.08:
                        current_col.append(x)
                    else:
                        columns.append((min(current_col), max(current_col)))
                        current_col = [x]
                if current_col:
                    columns.append((min(current_col), max(current_col)))
            
            print(f"📊 Detected numeric columns: {len(columns)}")
            for i, (x_min, x_max) in enumerate(columns):
                col_tokens = [p for p in numeric_tokens if x_min <= p["center_x"] <= x_max]
                if col_tokens:
                    avg_x = sum(p["center_x"] for p in col_tokens) / len(col_tokens)
                    sample_text = col_tokens[0]["text"] if col_tokens else "N/A"
                    print(f"   Column {i+1}: x={int(avg_x):4d}px | {len(col_tokens)} tokens | Sample: {sample_text}")
            
            if columns:
                rightmost_col = columns[-1]
                rightmost_tokens = [p for p in numeric_tokens if rightmost_col[0] <= p["center_x"] <= rightmost_col[1]]
                if rightmost_tokens:
                    print(f"\n➡️  RIGHT-MOST COLUMN (x={int(rightmost_col[0])}-{int(rightmost_col[1])}px):")
                    for token in sorted(rightmost_tokens, key=lambda x: x["center_y"])[-5:]:
                        amount = self._parse_currency(token["text"])
                        if amount > 0:
                            print(f"   ${amount:>10.2f} @ y={int(token['center_y']):4d}")
        
        # Check for address labels
        address_labels = ["bill", "ship", "attention", "attn", "to:", "address"]
        for pred in predictions:
            text_lower = pred["text"].lower()
            if any(label in text_lower for label in address_labels):
                print(f"📍 Address label: '{pred['text']}' @ ({int(pred['center_x'])}, {int(pred['center_y'])})")
        
        print("="*70)
    
    def _extract_invoice_complete(self, predictions: List[Dict], img_w: int, img_h: int) -> Dict:
        """COMPLETE EXTRACTION PIPELINE - FIXED VERSION"""
        result = self._empty_result("", img_w, img_h)["structured_data"]
        
        # Vendor (top-left) - FIXED: Filter out address labels
        result["vendor_company"] = self._extract_vendor_company(predictions, img_w, img_h)
        
        # Invoice number
        result["invoice_number"] = self._extract_invoice_number(predictions, img_w, img_h)
        
        # Dates
        dates = self._extract_dates(predictions)
        if dates:
            result["invoice_date"] = dates[0]
            if len(dates) > 1:
                result["due_date"] = dates[1]
        
        # Order ID (expanded)
        result["order_id"] = self._extract_order_id_comprehensive(predictions, img_w, img_h)
        
        # Addresses - FIXED: Better boundary detection
        addresses = self._extract_addresses_with_boundaries(predictions, img_w, img_h)
        result["bill_to"] = addresses.get("bill_to", "")
        result["ship_to"] = addresses.get("ship_to", "")

        # Line items (table detection)
        result["items"] = self._extract_line_items_table(predictions, img_w, img_h)
        
        # Financials - FIXED: Stricter label verification
        financials = self._extract_financial_strict(predictions, img_w, img_h)
        result.update(financials)
        
        # Payment terms
        result["payment_terms"] = self._extract_payment_terms_improved(predictions, img_w, img_h)
        
        # Notes - NEW FIELD
        result["notes"] = self._extract_notes(predictions, img_h)
        
        # Currency
        result["currency"] = self._detect_currency(predictions)
        
        # Confidence
        entity_confs = [p["confidence"] for p in predictions if p["label"] != "O"]
        if entity_confs:
            result["extraction_confidence"] = sum(entity_confs) / len(entity_confs)
        
        return result
    
    def _extract_vendor_company(self, predictions: List[Dict], img_w: int, img_h: int) -> str:
        """Extract vendor company using BETTER LOGIC, not character correction"""
        # 1. First look for HEADER labeled tokens (most reliable)
        header_tokens = [
            p for p in predictions 
            if p["label"] in ["B-HEADER", "I-HEADER"] 
            and p["center_y"] < img_h * 0.3  # Top area
        ]
        
        if header_tokens:
            # Group by line (similar Y position)
            lines = {}
            line_tol = img_h * 0.02
            
            for token in header_tokens:
                y = token["center_y"]
                found = False
                for line_y in lines:
                    if abs(y - line_y) < line_tol:
                        lines[line_y].append(token)
                        found = True
                        break
                if not found:
                    lines[y] = [token]
            
            # Process the highest header line (topmost)
            if lines:
                top_line_y = min(lines.keys())
                top_tokens = sorted(lines[top_line_y], key=lambda x: x["center_x"])
                
                # Check if this looks like a vendor name (not invoice label, etc.)
                line_text = " ".join(t["text"] for t in top_tokens)
                if self._is_likely_vendor_name(line_text):
                    print(f"🔍 Header vendor: '{line_text}'")
                    return line_text
        
        # 2. Look for the largest text in top-left quadrant (fallback)
        top_left = [
            p for p in predictions 
            if p["center_y"] < img_h * 0.2 and p["center_x"] < img_w * 0.4
            and not self._is_address_label(p["text"])
            and not self._looks_like_date(p["text"])
            and not p.get("is_numeric")
        ]
        
        if not top_left:
            return ""
        
        # Sort by size (area = width * height) - vendors are often largest
        top_left.sort(key=lambda x: -(x["width"] * x["height"]))
        
        for token in top_left[:3]:  # Check top 3 largest
            text = token["text"].strip()
            
            # Skip obvious non-vendors
            if (len(text) < 3 or 
                text.lower() in ["invoice", "date", "order", "page", "no"] or
                any(c.isdigit() for c in text)):
                continue
            
            # Check if text contains likely vendor patterns
            if self._is_likely_vendor_name(text):
                print(f"🔍 Largest text vendor: '{text}'")
                return text
            
            # If text is split across tokens, try to find adjacent tokens
            # Look for tokens on same line within reasonable distance
            same_line_tokens = [
                p for p in predictions
                if abs(p["center_y"] - token["center_y"]) < img_h * 0.02
                and p["center_x"] > token["center_x"]
                and p["center_x"] - token["center_x"] < img_w * 0.2
            ]
            
            if same_line_tokens:
                same_line_tokens.sort(key=lambda x: x["center_x"])
                combined = text + " " + " ".join(t["text"] for t in same_line_tokens)
                if self._is_likely_vendor_name(combined):
                    print(f"🔍 Combined vendor: '{combined}'")
                    return combined
        
        return ""


    def _is_likely_vendor_name(self, text: str) -> bool:
        """Determine if text looks like a vendor name"""
        if not text or len(text) < 3:
            return False
        
        text_lower = text.lower()
        
        # Definitely NOT vendor names
        not_vendor_patterns = [
            r'^invoice\s*(?:no|#|number|date)?',
            r'^order\s*(?:no|#|id)?',
            r'^bill\s+to',
            r'^ship\s+to',
            r'^address',
            r'^phone',
            r'^email',
            r'^page\s*\d+',
            r'^\d+[/-]\d+[/-]\d+',  # Dates
            r'^[\d\s\$\€\£\¥\₹]+$',  # Just numbers/currency
        ]
        
        for pattern in not_vendor_patterns:
            if re.match(pattern, text_lower):
                return False
        
        # Vendor names SHOULD have these characteristics:
        # 1. Not all numbers
        if re.fullmatch(r'[\d\s]+', text):
            return False
        
        # 2. Not just currency amounts
        if re.match(r'^[\$\€\£\¥\₹]?\s*\d+\.?\d*$', text):
            return False
        
        # 3. Not too many numbers (maybe 30% max)
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count > len(text) * 0.3:
            return False
        
        # 4. Usually has letters
        if not re.search(r'[A-Za-z]', text):
            return False
        
        # 5. Usually not all uppercase (might be a header label)
        # But some vendors ARE all caps, so this is weak
        
        return True
    
    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date"""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)
    
    def _extract_addresses_with_boundaries(self, predictions: List[Dict], img_w: int, img_h: int) -> Dict:
        """Simpler address extraction with strict limits"""
        result = {"bill_to": "", "ship_to": ""}
        
        # Sort predictions by Y position to process in reading order
        sorted_predictions = sorted(predictions, key=lambda x: (x["center_y"], x["center_x"]))
        
        for pred in sorted_predictions:
            text_lower = pred["text"].lower().strip()
            
            # BILL TO - FIXED: Better label matching
            if "bill" in text_lower:
                # Look for the actual address content
                address_text = self._extract_address_content(pred, sorted_predictions, img_w, img_h)
                if address_text:
                    result["bill_to"] = address_text
                    continue  # Don't break, continue to look for ship_to
            
            # SHIP TO - FIXED: Better label matching  
            if "ship" in text_lower and not result["ship_to"]:  # Only set if not already found
                # Look for the actual address content
                address_text = self._extract_address_content(pred, sorted_predictions, img_w, img_h)
                if address_text:
                    result["ship_to"] = address_text
        
        return result


    def _extract_addresses_with_boundaries(self, predictions: List[Dict], img_w: int, img_h: int) -> Dict:
        """Address extraction with column awareness"""
        result = {"bill_to": "", "ship_to": ""}
        
        # First, find ALL address labels and their positions
        bill_labels = []
        ship_labels = []
        
        for pred in predictions:
            text_lower = pred["text"].lower().strip()
            if "bill" in text_lower:
                bill_labels.append(pred)
            elif "ship" in text_lower:
                ship_labels.append(pred)
        
        # If we found both labels, determine columns and extract accordingly
        if bill_labels and ship_labels:
            # Take the most likely label (first one found in each category)
            bill_label = bill_labels[0]
            ship_label = ship_labels[0]
            
            # Determine which is left, which is right
            if bill_label["center_x"] < ship_label["center_x"]:
                # Normal case: Bill To is left, Ship To is right
                left_label, right_label = bill_label, ship_label
                left_field, right_field = "bill_to", "ship_to"
            else:
                # Reversed: Ship To is left, Bill To is right
                left_label, right_label = ship_label, bill_label
                left_field, right_field = "ship_to", "bill_to"
            
            # Calculate midpoint between columns for better separation
            midpoint_x = (left_label["center_x"] + right_label["center_x"]) / 2
            
            # Extract left column (strict left of midpoint)
            left_content = self._extract_address_content_fixed(
                left_label, predictions, img_w, img_h, 
                max_x=midpoint_x - (img_w * 0.05),  # Leave small gap
                side="left"
            )
            
            # Extract right column (strict right of midpoint)
            right_content = self._extract_address_content_fixed(
                right_label, predictions, img_w, img_h,
                min_x=midpoint_x + (img_w * 0.05),  # Leave small gap
                side="right"
            )
            
            result[left_field] = left_content
            result[right_field] = right_content
        
        # Fallback: If only one label found, use original method
        elif bill_labels:
            result["bill_to"] = self._extract_address_content_fixed(
                bill_labels[0], predictions, img_w, img_h, side="unknown"
            )
        elif ship_labels:
            result["ship_to"] = self._extract_address_content_fixed(
                ship_labels[0], predictions, img_w, img_h, side="unknown"
            )
        
        return result

    def _extract_address_content_fixed(self, label_pred: Dict, predictions: List[Dict], 
                                    img_w: int, img_h: int, 
                                    max_x: float = None, min_x: float = None,
                                    side: str = "unknown") -> str:
        """Extract address content - SIMPLIFIED FIX"""
        candidates = []
        
        # Horizontal tolerance settings
        if side == "left":
            horiz_tolerance_left = img_w * 0.15
            horiz_tolerance_right = img_w * 0.05
        elif side == "right":
            horiz_tolerance_left = img_w * 0.05
            horiz_tolerance_right = img_w * 0.15
        else:
            horiz_tolerance_left = img_w * 0.1
            horiz_tolerance_right = img_w * 0.1
        
        # Simple table header filter
        skip_tokens = {"item", "qty", "quantity", "description", "price", "rate"}
        
        for other in predictions:
            # Skip if above or same level as label
            if other["center_y"] <= label_pred["center_y"]:
                continue
            
            # Shorter vertical range
            if other["center_y"] > label_pred["center_y"] + (img_h * 0.15):
                continue
            
            # Horizontal check
            offset = other["center_x"] - label_pred["center_x"]
            if offset < 0 and abs(offset) > horiz_tolerance_left:
                continue
            if offset >= 0 and offset > horiz_tolerance_right:
                continue
            
            # Column boundaries
            if max_x and other["center_x"] > max_x:
                continue
            if min_x and other["center_x"] < min_x:
                continue
            
            # Skip address labels
            if self._is_address_label(other["text"]):
                continue
            
            # Skip table headers
            if other["text"].lower().strip() in skip_tokens:
                continue
            
            # Skip ALL CAPS short words (often headers)
            if other["text"].isupper() and len(other["text"]) < 8:
                continue
            
            candidates.append(other)
        
        if not candidates:
            return ""
        
        # Simple address building
        return self._build_simple_address(candidates, img_h)

    def _build_simple_address(self, candidates: List[Dict], img_h: int) -> str:
        """Build address - simple version"""
        candidates.sort(key=lambda x: x["center_y"])
        
        # Group by line
        lines = {}
        for token in candidates:
            y = token["center_y"]
            line_found = False
            
            for line_y in lines.keys():
                if abs(y - line_y) < img_h * 0.015:
                    lines[line_y].append(token)
                    line_found = True
                    break
            
            if not line_found:
                lines[y] = [token]
        
        # Build lines
        address_lines = []
        for line_y in sorted(lines.keys()):
            tokens = sorted(lines[line_y], key=lambda x: x["center_x"])
            line_text = " ".join(t["text"] for t in tokens).strip()
            
            if line_text:
                address_lines.append(line_text)
            
            if len(address_lines) >= 3:
                break
        
        return "\n".join(address_lines) if address_lines else ""

    def _build_address_from_candidates(self, candidates: List[Dict], img_h: int) -> str:
        """Build address string from candidate tokens"""
        # Sort by Y position
        candidates.sort(key=lambda x: x["center_y"])
        
        # Group by line
        lines = {}
        line_tolerance = img_h * 0.015
        
        for token in candidates:
            y_pos = token["center_y"]
            found_line = False
            
            for line_y in lines.keys():
                if abs(y_pos - line_y) < line_tolerance:
                    lines[line_y].append(token)
                    found_line = True
                    break
            
            if not found_line:
                lines[y_pos] = [token]
        
        # Build address lines
        address_lines = []
        for line_y in sorted(lines.keys()):
            line_tokens = sorted(lines[line_y], key=lambda x: x["center_x"])
            line_text = " ".join(t["text"] for t in line_tokens).strip()
            
            if line_text:
                address_lines.append(line_text)
            
            if len(address_lines) >= 3:
                break
        
        return "\n".join(address_lines) if address_lines else ""

    def _extract_address_content(self, label_pred: Dict, predictions: List[Dict], img_w: int, img_h: int) -> str:
        """Deprecated - use _extract_address_content_fixed instead"""
        return self._extract_address_content_fixed(label_pred, predictions, img_w, img_h, side="unknown")


    def _is_address_label(self, text: str) -> bool:
        """Check if text is an address label (not actual address content) - FIXED"""
        text_lower = text.lower().strip()
        
        # Common address labels
        label_keywords = ["bill", "ship", "to:", "attention", "attn", "address", "from", "for"]
        
        # Check if it's likely a label
        for keyword in label_keywords:
            if keyword in text_lower:
                # Additional checks to avoid false positives
                words = text.split()
                
                # Labels are usually short (1-3 words)
                if len(words) <= 3:
                    # Check for common label patterns
                    if (text_lower.endswith(":") or 
                        text_lower.endswith("to") or
                        ":" in text_lower or
                        text_lower in ["bill", "ship", "attention", "attn"]):
                        return True
        
        return False
    
    def _extract_financial_strict(self, predictions: List[Dict], img_w: int, img_h: int) -> Dict:
        """Strict financial extraction with label verification"""
        financials = {
            "subtotal": 0.0, "shipping": 0.0, "tax": 0.0, 
            "discount": 0.0, "total_amount": 0.0, "balance_due": 0.0
        }
        
        print(f"\n💰 STRICT FINANCIAL EXTRACTION")
        
        # Field patterns with strict matching
        field_patterns = [
            ("subtotal", ["sub.?total", "amount before tax", "merchandise", "product total"]),
            ("shipping", ["shipping", "freight", "delivery", "handling", "ship"]),
            ("tax", ["tax", "vat", "gst", "sales tax", "taxable"]),
            ("discount", ["discount", "less", "deduction", "adjustment", "promo"]),
            ("total_amount", ["total", "amount due", "balance due", "net due", "payable", "grand total"]),
            ("balance_due", ["balance due", "amount due", "payable", "net due"]),
        ]
        
        # First pass: Find all potential labels
        all_labels = []
        for pred in predictions:
            text_lower = pred["text"].lower()
            for field_name, patterns in field_patterns:
                if any(re.search(pattern, text_lower) for pattern in patterns):
                    all_labels.append((field_name, pred))
        
        # Group labels by Y position (rows)
        rows = {}
        row_tolerance = img_h * 0.02
        
        for field_name, label in all_labels:
            row_found = False
            for row_y in rows.keys():
                if abs(label["center_y"] - row_y) < row_tolerance:
                    rows[row_y].append((field_name, label))
                    row_found = True
                    break
            if not row_found:
                rows[label["center_y"]] = [(field_name, label)]
        
        # For each row, find the amount to the right
        for row_y, row_labels in rows.items():
            # Find numeric tokens to the right of labels
            for field_name, label in row_labels:
                # Look for amount in same row (similar Y)
                candidate_amounts = []
                for pred in predictions:
                    if (pred.get("is_numeric") and
                        pred["center_x"] > label["center_x"] and
                        abs(pred["center_y"] - row_y) < row_tolerance * 2 and
                        pred["center_x"] < label["center_x"] + img_w * 0.4):
                        
                        amount = self._parse_currency(pred["text"])
                        if amount > 0:
                            candidate_amounts.append((amount, pred))
                
                # Take closest amount to the right
                if candidate_amounts:
                    candidate_amounts.sort(key=lambda x: abs(x[1]["center_x"] - label["center_x"]))
                    best_amount, best_token = candidate_amounts[0]
                    
                    # Only update if we don't have this field yet or this is a better match
                    if financials[field_name] == 0.0 or field_name == "total_amount":
                        financials[field_name] = best_amount
                        print(f"   ✅ {field_name:15} = ${best_amount:.2f} | Label: '{label['text']}'")
        
        # If total not found, try bottom 15% of invoice
        if financials["total_amount"] == 0.0:
            bottom_y = img_h * 0.85
            bottom_amounts = []
            for pred in predictions:
                if pred.get("is_numeric") and pred["center_y"] > bottom_y:
                    amount = self._parse_currency(pred["text"])
                    if amount > 100:  # Total should be significant
                        bottom_amounts.append((amount, pred))
            
            if bottom_amounts:
                bottom_amounts.sort(key=lambda x: x[0], reverse=True)  # Largest amount
                financials["total_amount"] = bottom_amounts[0][0]
                print(f"   ⚠️  total_amount   = ${bottom_amounts[0][0]:.2f} (inferred from bottom)")
        
        # If shipping looks wrong (too high, might be line item), set to 0
        if financials["shipping"] > financials["total_amount"] * 0.5:  # Shipping > 50% of total is suspicious
            print(f"   ⚠️  shipping amount ${financials['shipping']:.2f} looks wrong, setting to 0")
            financials["shipping"] = 0.0
        
        return financials
    
    def _extract_order_id_comprehensive(self, predictions: List[Dict], img_w: int, img_h: int) -> str:
        """Fixed Order ID extraction with proper spatial relationships"""
        
        # Step 1: Find the EXACT "Order ID:" label
        order_label_tokens = []
        
        for pred in predictions:
            text_lower = pred["text"].lower().strip()
            
            # Look for exact "Order ID" or similar patterns
            if (text_lower == "order" or 
                text_lower == "order id" or 
                text_lower == "order id:" or
                text_lower.startswith("order id")):
                order_label_tokens.append(pred)
        
        print(f"🔍 Found {len(order_label_tokens)} Order ID label tokens")
        
        # Step 2: For each Order ID label, find the value to its RIGHT on the SAME line
        for label_token in order_label_tokens:
            print(f"  Checking Order ID label: '{label_token['text']}' at ({int(label_token['center_x'])}, {int(label_token['center_y'])})")
            
            # Look for tokens on the SAME line (strict vertical tolerance)
            same_line_tokens = []
            line_tolerance = img_h * 0.01  # Very strict: only 1% height difference
            
            for pred in predictions:
                if pred is label_token:
                    continue
                    
                # Check if on EXACT same line
                if abs(pred["center_y"] - label_token["center_y"]) < line_tolerance:
                    same_line_tokens.append(pred)
            
            if not same_line_tokens:
                print(f"    No tokens on same line found")
                continue
            
            # Sort by X position (left to right)
            same_line_tokens.sort(key=lambda x: x["center_x"])
            
            # Find the Order ID label position in the line
            line_tokens = same_line_tokens.copy()
            line_tokens.append(label_token)
            line_tokens.sort(key=lambda x: x["center_x"])
            
            # Find tokens to the RIGHT of the Order ID label
            tokens_to_right = []
            for token in line_tokens:
                if token["center_x"] > label_token["center_x"]:
                    tokens_to_right.append(token)
            
            print(f"    Found {len(tokens_to_right)} tokens to the right of label")
            
            # Step 3: Extract the value (first meaningful token to the right)
            if tokens_to_right:
                # The value is usually the first token to the right
                first_right = tokens_to_right[0]
                value = first_right["text"].strip()
                
                # Clean up common separators
                if value.startswith(":") or value.startswith("#"):
                    value = value[1:].strip()
                
                # Check if this looks like an Order ID
                if self._is_valid_order_id(value):
                    print(f"✅ Order ID found: '{value}'")
                    return value
                
                # Sometimes the value might be in multiple tokens (e.g., "CA-2012-AT10735140-41150" split)
                # Try combining tokens to the right
                combined_value = ""
                for token in tokens_to_right[:5]:  # Combine up to 5 tokens to the right
                    token_text = token["text"].strip()
                    
                    # Skip separators
                    if token_text in [":", "#", "-", ":", ";"]:
                        continue
                        
                    # Check if this token looks like part of an Order ID
                    if re.match(r'^[A-Z0-9\-]+$', token_text):
                        if combined_value:
                            combined_value += " " + token_text
                        else:
                            combined_value = token_text
                    else:
                        break  # Stop if we hit non-ID text
                
                if combined_value and self._is_valid_order_id(combined_value):
                    print(f"✅ Order ID (combined): '{combined_value}'")
                    return combined_value
        
        # Step 4: Fallback - Look for Order ID patterns with context
        print("⚠️  Trying fallback pattern matching...")
        
        # Improved patterns that match "Order ID: VALUE"
        patterns = [
            r'(?:order\s*id|Order\s*ID)[\s:]*([A-Z0-9\-]{6,})',
            r'(?:order\s*id|Order\s*ID)[\s:]*([A-Z]{2}-\d{4}-[A-Z0-9\-]+)',  # CA-2012-AT10735140-41150
            r'Order\s*ID\s*[:#]\s*([A-Z0-9\-]{6,})',
            r'(?:PO|P\.?O\.?)[\s:]*([A-Z0-9\-]{6,})',
            r'#\s*([A-Z0-9\-]{6,})',
        ]
        
        # Group tokens by very tight lines
        lines = {}
        tight_tolerance = img_h * 0.008  # 0.8% height tolerance
        
        for pred in predictions:
            y_key = round(pred["center_y"] / tight_tolerance)
            if y_key not in lines:
                lines[y_key] = []
            lines[y_key].append(pred)
        
        # Check each line
        for line_key in sorted(lines.keys()):
            tokens = sorted(lines[line_key], key=lambda x: x["center_x"])
            combined_line = " ".join(t["text"] for t in tokens)
            
            for pattern in patterns:
                match = re.search(pattern, combined_line, re.IGNORECASE)
                if match:
                    order_id = match.group(1).strip()
                    if self._is_valid_order_id(order_id):
                        print(f"✅ Order ID (line pattern): '{order_id}' from line: '{combined_line[:50]}...'")
                        return order_id
        
        return ""

    def _is_valid_order_id(self, text: str) -> bool:
        """Strict validation for Order IDs"""
        if not text or len(text) < 6:  # Order IDs are usually at least 6 chars
            return False
        
        # Remove common separators and whitespace
        clean = re.sub(r'[\s:;#]', '', text)
        
        # Common false positives
        false_positives = ["invoice", "date", "due", "total", "amount", "terms", "payment"]
        if clean.lower() in false_positives:
            return False
        
        # Check if it looks like a date
        if self._looks_like_date(clean):
            return False
        
        # Check if it looks like a currency amount
        if re.match(r'^[\$\€\£]?\d+\.?\d*$', clean):
            return False
        
        # Order ID patterns
        patterns = [
            r'^[A-Z]{2,3}-\d{4}-[A-Z0-9\-]+$',  # CA-2012-AT10735140-41150
            r'^[A-Z]{2,}\d{6,}$',  # ABC123456
            r'^\d{6,}$',  # 123456789
            r'^[A-Z0-9]{6,}-[A-Z0-9\-]+$',  # Mixed with dashes
            r'^PO\d{6,}$',  # PO123456
        ]
        
        for pattern in patterns:
            if re.match(pattern, clean, re.IGNORECASE):
                return True
        
        # If it has good mix of letters and numbers, and reasonable length
        letter_count = sum(1 for c in clean if c.isalpha())
        digit_count = sum(1 for c in clean if c.isdigit())
        
        if letter_count >= 2 and digit_count >= 3 and len(clean) >= 8:
            return True
        
        return False
    
    def _extract_notes(self, predictions: List[Dict], img_h: int) -> str:
        """Extract notes only - stop at Terms/Order ID sections"""
        notes_keywords = ["notes", "remarks", "comments"]
        
        # Find notes label
        notes_label = None
        for pred in predictions:
            text_lower = pred["text"].lower()
            if any(keyword in text_lower for keyword in notes_keywords):
                notes_label = pred
                break
        
        if not notes_label:
            return ""
        
        # Group all tokens by line
        all_lines = {}
        line_tolerance = img_h * 0.02
        
        for pred in predictions:
            y = pred["center_y"]
            line_found = False
            
            for line_y in all_lines.keys():
                if abs(y - line_y) < line_tolerance:
                    all_lines[line_y].append(pred)
                    line_found = True
                    break
            
            if not line_found:
                all_lines[y] = [pred]
        
        # Sort lines by Y position
        sorted_lines = sorted(all_lines.items(), key=lambda x: x[0])
        
        # Find the line with the notes label
        notes_line_idx = -1
        for idx, (line_y, tokens) in enumerate(sorted_lines):
            for token in tokens:
                if token is notes_label:  # Compare by object reference
                    notes_line_idx = idx
                    break
            if notes_line_idx != -1:
                break
        
        if notes_line_idx == -1:
            return ""
        
        # Collect note lines starting from line after the notes label
        notes_lines = []
        section_starters = ["terms", "order", "ship", "bill", "payment", "due", "attention"]
        
        for i in range(notes_line_idx + 1, min(notes_line_idx + 8, len(sorted_lines))):
            line_y, tokens = sorted_lines[i]
            
            # Sort tokens in line left to right
            tokens.sort(key=lambda x: x["center_x"])
            line_text = " ".join(t["text"] for t in tokens).strip()
            
            # Skip the "Notes:" label itself if it appears again
            if any(keyword in line_text.lower() for keyword in notes_keywords):
                continue
            
            # Stop if we hit a new section
            line_lower = line_text.lower()
            if any(starter in line_lower for starter in section_starters):
                break
            
            # Skip empty lines
            if line_text:
                notes_lines.append(line_text)
        
        # Join and return
        return "\n".join(notes_lines)
    
    def _extract_payment_terms_improved(self, predictions: List[Dict], img_w: int, img_h: int) -> str:
        """Improved payment terms extraction using spatial positioning"""
        # Payment term label patterns
        payment_labels = [
            "payment terms", "terms", "net", "due", "payable", "balance due",
            "terms:", "payment:", "net due", "due date", "ship mode", "ship terms"
        ]
        
        # First, look for "Ship Mode:" to know where to search
        ship_mode_token = None
        for pred in predictions:
            if "ship mode" in pred["text"].lower():
                ship_mode_token = pred
                break
        
        # If we found "Ship Mode:", search near it (same row or below)
        if ship_mode_token:
            # Look for payment terms in same area as Ship Mode
            search_area_top = ship_mode_token["center_y"] - (img_h * 0.05)
            search_area_bottom = ship_mode_token["center_y"] + (img_h * 0.15)
            
            # Get tokens in this area
            area_tokens = [
                p for p in predictions
                if search_area_top < p["center_y"] < search_area_bottom
            ]
            
            # Look for payment term labels in this area
            for pred in area_tokens:
                text_lower = pred["text"].lower()
                
                # Check if token is a payment term label
                is_label = any(label in text_lower for label in payment_labels)
                
                if is_label:
                    # Look for value to the RIGHT of this label
                    value = self._extract_value_right_of_label(pred, area_tokens, img_w, img_h)
                    if value:
                        return value
        
        # Fallback: Search entire invoice, focusing on right side
        # Payment terms are often on the right side of invoices
        right_side_tokens = [p for p in predictions if p["center_x"] > img_w * 0.6]
        
        for pred in right_side_tokens:
            text_lower = pred["text"].lower()
            
            # Check if token is a payment term label
            is_label = any(label in text_lower for label in payment_labels)
            
            if is_label:
                # Extract value to the right
                value = self._extract_value_right_of_label(pred, predictions, img_w, img_h)
                if value:
                    # Clean up the value - remove the label if it's included
                    for label in payment_labels:
                        if label in value.lower():
                            # Try to extract just the value part
                            parts = re.split(r':|[-–]', value, maxsplit=1)
                            if len(parts) > 1:
                                return parts[1].strip()
                    return value
        
        # Legacy pattern matching (fallback)
        term_patterns = [
            r'(?:payment\s+terms|terms|net|balance\s+due)[:\s-]*([^\n]+)',
            r'(?:net\s+)(\d+)\s*(?:days?|d)',
            r'(?:due\s+in\s+)(\d+)\s*days',
            r'(\d+)\s*%\s*\d+\s*days',
            r'(?:COD|Cash\s+on\s+Delivery)',
            r'(?:due\s+(?:upon|on)\s+(?:receipt|delivery|invoice))',
        ]
        
        for pred in predictions:
            text = pred["text"]
            for pattern in term_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Return the CAPTURED GROUP (value), not the entire text
                    if match.groups():
                        return match.group(1).strip()
                    # If no capture group, return the full match
                    return match.group(0).strip()
        
        return ""

    def _extract_value_right_of_label(self, label_token: Dict, predictions: List[Dict], img_w: int, img_h: int) -> str:
        """Extract value to the right of a label on the same row"""
        
        # Find tokens on the same row (similar Y position)
        row_tolerance = img_h * 0.02
        same_row_tokens = [
            p for p in predictions
            if abs(p["center_y"] - label_token["center_y"]) < row_tolerance
            and p["center_x"] > label_token["center_x"]
            and p["center_x"] < label_token["center_x"] + (img_w * 0.3)  # Within 30% to the right
        ]
        
        if not same_row_tokens:
            return ""
        
        # Sort by X position (left to right)
        same_row_tokens.sort(key=lambda x: x["center_x"])
        
        # Take the first token to the right (closest to label)
        first_right_token = same_row_tokens[0]
        
        # Check if this token looks like a value (not another label)
        text_lower = first_right_token["text"].lower()
        payment_labels = ["payment terms", "terms", "net", "due", "payable", "balance due"]
        
        # If it's another label, skip it and look further right
        if any(label in text_lower for label in payment_labels):
            if len(same_row_tokens) > 1:
                return same_row_tokens[1]["text"]
            return ""
        
        return first_right_token["text"]
    
    def _extract_line_items_table(self, predictions: List[Dict], img_w: int, img_h: int) -> List[Dict]:
        """Robust line item extraction using spatial table detection"""
        print(f"🔍 Starting robust line item extraction...")
        
        items = []
        
        # Step 1: Find the item section by looking for table structure
        # Look for "Item" or "Description" header
        item_header = None
        for pred in predictions:
            text_lower = pred["text"].lower()
            if text_lower in ["item", "description", "product", "part"]:
                item_header = pred
                break
        
        # Step 2: Find quantity/rate/amount headers
        qty_header = None
        rate_header = None
        amt_header = None
        qty_keywords = ["qty", "quantity", "qty."]
        rate_keywords = ["rate", "price", "unit", "each"]
        amt_keywords = ["amount", "total", "amt", "extended"]
        
        for pred in predictions:
            text_lower = pred["text"].lower()
            if any(kw in text_lower for kw in qty_keywords):
                qty_header = pred
            elif any(kw in text_lower for kw in rate_keywords):
                rate_header = pred
            elif any(kw in text_lower for kw in amt_keywords):
                amt_header = pred
        
        # Step 3: If we found headers, use them to define table structure
        if qty_header or rate_header or amt_header:
            print("✅ Found table headers - using table detection method")
            items = self._extract_items_with_table_structure(predictions, img_w, img_h, 
                                                            item_header, qty_header, rate_header, amt_header)
        
        # Step 4: If no headers found, use spatial clustering
        if not items:
            print("⚠️  No table headers found - using spatial clustering method")
            items = self._extract_items_with_spatial_clustering(predictions, img_w, img_h)
        
        # Step 5: If still no items, use pattern-based extraction
        if not items:
            print("⚠️  No items with spatial clustering - using pattern matching")
            items = self._extract_items_with_pattern_matching(predictions, img_w, img_h)
        
        print(f"📦 Extracted {len(items)} line items")
        return items

    def _extract_items_with_table_structure(self, predictions: List[Dict], img_w: int, img_h: int,
                                            item_header, qty_header, rate_header, amt_header):
        """Extract items using detected table structure with multi-line description support"""
        items = []
        
        # Determine table boundaries
        table_top = img_h * 0.3
        table_bottom = img_h * 0.7
        
        if item_header:
            table_top = item_header["center_y"] + (img_h * 0.02)
        
        # Look for summary section to determine table bottom
        summary_keywords = ["subtotal", "total", "balance", "shipping", "tax"]
        for pred in predictions:
            if any(kw in pred["text"].lower() for kw in summary_keywords):
                if pred["center_y"] > table_top:
                    table_bottom = min(table_bottom, pred["center_y"] - (img_h * 0.02))
                    break
        
        # Get all tokens in table region
        table_tokens = [p for p in predictions if table_top < p["center_y"] < table_bottom]
        
        if not table_tokens:
            return []
        
        # Step 1: Find numeric columns
        numeric_tokens = [t for t in table_tokens if t.get("is_numeric")]
        
        if not numeric_tokens:
            return []
        
        # Cluster numeric tokens by X position (columns)
        columns = self._cluster_by_x_position(numeric_tokens, img_w * 0.05)
        
        print(f"   Found {len(columns)} numeric columns")
        
        # NEW: Group tokens by Y position to find rows
        all_row_tokens = table_tokens.copy()
        rows = self._cluster_by_y_position(all_row_tokens, img_h * 0.02)
        
        print(f"   Found {len(rows)} potential rows")
        
        # NEW: Identify which rows contain numeric values (actual item rows)
        item_rows = []
        for row_y, row_tokens in sorted(rows.items()):
            row_numerics = [t for t in row_tokens if t.get("is_numeric")]
            if row_numerics:
                item_rows.append({
                    'y': row_y,
                    'tokens': row_tokens,
                    'numerics': row_numerics
                })
        
        print(f"   Found {len(item_rows)} item rows with numeric values")
        
        # Process each item row
        for item_row in item_rows:
            row_tokens = item_row['tokens']
            row_numerics = item_row['numerics']
            
            # Sort tokens in row by X position
            row_tokens.sort(key=lambda x: x["center_x"])
            
            # Extract components
            description_parts = []
            quantity = 0
            rate = 0.0
            amount = 0.0
            
            # NEW: Look for multi-line descriptions
            # Get the description token(s) in this row (non-numeric, longer text)
            row_description_tokens = [t for t in row_tokens 
                                    if not t.get("is_numeric") 
                                    and len(t["text"]) > 3]
            
            if row_description_tokens:
                # Add the main description from this row
                main_desc_token = row_description_tokens[0]  # Usually the first non-numeric
                description_parts.append(main_desc_token["text"])
                
                # NEW: Look for continuation lines (tokens directly below this one)
                # Check if there are tokens directly below this description that might be continuation
                desc_bottom_y = main_desc_token["center_y"] + (img_h * 0.02)
                
                # Find tokens that are aligned with this description (similar X position)
                # and are below it but not too far (within typical line spacing)
                continuation_tokens = []
                for y_key, other_row_tokens in rows.items():
                    # Skip if it's the same row
                    if y_key == item_row['y']:
                        continue
                    
                    # Check if this row is below our current row
                    if y_key > item_row['y']:
                        # Check if any tokens in this row are aligned with our description
                        for token in other_row_tokens:
                            # Check X alignment (within description column width)
                            x_diff = abs(token["center_x"] - main_desc_token["center_x"])
                            if (x_diff < img_w * 0.1 and  # Aligned horizontally
                                not token.get("is_numeric") and  # Not numeric
                                len(token["text"]) > 2):  # Meaningful text
                                
                                # Also check that this token isn't part of another item
                                # (no numeric values in its row)
                                other_row_numerics = [t for t in other_row_tokens if t.get("is_numeric")]
                                if not other_row_numerics:
                                    continuation_tokens.append(token)
                                    # Break after finding first continuation token in this row
                                    break
                
                # Sort continuation tokens by Y position and add to description
                if continuation_tokens:
                    continuation_tokens.sort(key=lambda x: x["center_y"])
                    for cont_token in continuation_tokens:
                        description_parts.append(cont_token["text"])
            
            # Parse numeric values (same as before)
            if len(row_numerics) >= 3:
                # Likely: Qty, Rate, Amount
                quantity = self._parse_quantity(row_numerics[0]["text"])
                rate = self._parse_currency(row_numerics[1]["text"])
                amount = self._parse_currency(row_numerics[2]["text"])
            elif len(row_numerics) == 2:
                # Determine which is which by position and value
                val1 = self._parse_currency(row_numerics[0]["text"])
                val2 = self._parse_currency(row_numerics[1]["text"])
                
                # Heuristic: smaller whole number is quantity
                if val1 < 100 and val1 == int(val1):
                    quantity = int(val1)
                    amount = val2
                    if quantity > 0 and amount > 0:
                        rate = amount / quantity
                else:
                    # Might be rate and amount
                    rate = val1
                    amount = val2
            
            # Combine description parts
            if description_parts:
                description = " ".join(description_parts)
                
                # Validate this looks like a real item
                if self._is_valid_item_description(description):
                    items.append({
                        "description": description,
                        "quantity": quantity,
                        "rate": rate,
                        "amount": amount,
                    })
                    print(f"   ✅ Table item: '{description[:40]}...' (Qty: {quantity}, Amt: ${amount:.2f})")
        
        return items

    def _extract_items_with_spatial_clustering(self, predictions: List[Dict], img_w: int, img_h: int):
        """Extract items by clustering tokens spatially"""
        items = []
        
        # Filter out non-item tokens
        item_tokens = []
        non_item_keywords = ["bill", "ship", "date", "mode", "balance", "total", "subtotal", "shipping", "tax", "address"]
        
        for pred in predictions:
            text_lower = pred["text"].lower()
            is_non_item = any(kw in text_lower for kw in non_item_keywords)
            
            if not is_non_item and pred["center_y"] > img_h * 0.2 and pred["center_y"] < img_h * 0.8:
                item_tokens.append(pred)
        
        if not item_tokens:
            return []
        
        # Cluster tokens by Y position to find rows
        rows = self._cluster_by_y_position(item_tokens, img_h * 0.02)
        
        # Also cluster by X position to find columns
        all_tokens_sorted = sorted(item_tokens, key=lambda x: x["center_x"])
        columns = self._cluster_by_x_position(all_tokens_sorted, img_w * 0.05)
        
        print(f"   Spatial: Found {len(rows)} rows, {len(columns)} columns")
        
        # Process each row
        for row_y, row_tokens in sorted(rows.items()):
            if len(row_tokens) < 2:
                continue
            
            # Sort row tokens by X
            row_tokens.sort(key=lambda x: x["center_x"])
            
            # Skip if this looks like a header
            header_keywords = ["item", "qty", "quantity", "rate", "price", "amount", "total"]
            is_header = any(any(kw in t["text"].lower() for kw in header_keywords) for t in row_tokens[:3])
            if is_header:
                continue
            
            # Check for item characteristics
            has_text = any(not t.get("is_numeric") and len(t["text"]) > 3 for t in row_tokens)
            has_number = any(t.get("is_numeric") for t in row_tokens)
            
            if not (has_text and has_number):
                continue
            
            # Extract data
            description_parts = []
            numeric_values = []
            
            for token in row_tokens:
                if token.get("is_numeric"):
                    numeric_values.append(token)
                elif len(token["text"]) > 2:
                    description_parts.append(token["text"])
            
            if not description_parts:
                continue
            
            description = " ".join(description_parts)
            
            # Parse numeric values
            quantity = 0
            rate = 0.0
            amount = 0.0
            
            if numeric_values:
                numeric_values.sort(key=lambda x: x["center_x"])
                
                if len(numeric_values) >= 3:
                    quantity = self._parse_quantity(numeric_values[0]["text"])
                    rate = self._parse_currency(numeric_values[1]["text"])
                    amount = self._parse_currency(numeric_values[2]["text"])
                elif len(numeric_values) == 2:
                    val1 = self._parse_currency(numeric_values[0]["text"])
                    val2 = self._parse_currency(numeric_values[1]["text"])
                    
                    # Heuristic
                    if val1 < 100 and val1 == int(val1):
                        quantity = int(val1)
                        amount = val2
                    else:
                        amount = val1
                elif len(numeric_values) == 1:
                    val = self._parse_currency(numeric_values[0]["text"])
                    if val < 100 and val == int(val):
                        quantity = int(val)
                    else:
                        amount = val
            
            # Check if this is a valid item
            if self._is_valid_item_description(description):
                items.append({
                    "description": description,
                    "quantity": quantity,
                    "rate": rate,
                    "amount": amount,
                })
                print(f"   ✅ Spatial item: '{description[:40]}...'")
        
        return items

    def _extract_items_with_pattern_matching(self, predictions: List[Dict], img_w: int, img_h: int):
        """Extract items using pattern matching and contextual analysis"""
        items = []
        
        # Look for product-like patterns
        product_patterns = [
            # Multi-word patterns with mixed case
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z]',
            # Contains common product terms
            r'.*(box|storage|supply|item|product|part|sku).*',
            # Long text (likely description)
            r'^.{15,}$',
            # Contains model/sku patterns
            r'.*[A-Z]{2,}-\d+.*',
        ]
        
        # Find candidate descriptions
        candidate_descriptions = []
        
        for pred in predictions:
            text = pred["text"].strip()
            
            # Skip if too short or looks like metadata
            if len(text) < 8 or any(label in text.lower() for label in ["bill", "ship", "date", "total"]):
                continue
            
            # Check if text matches product patterns
            for pattern in product_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    candidate_descriptions.append(pred)
                    break
        
        # For each candidate description, find associated numeric values
        for desc_pred in candidate_descriptions:
            # Look for numeric values near this description
            search_top = desc_pred["center_y"] - (img_h * 0.05)
            search_bottom = desc_pred["center_y"] + (img_h * 0.1)
            search_left = desc_pred["center_x"]
            search_right = desc_pred["center_x"] + (img_w * 0.5)
            
            nearby_numerics = []
            for pred in predictions:
                if (pred.get("is_numeric") and
                    search_top < pred["center_y"] < search_bottom and
                    search_left < pred["center_x"] < search_right):
                    nearby_numerics.append(pred)
            
            if not nearby_numerics:
                continue
            
            # Sort numerics by X position
            nearby_numerics.sort(key=lambda x: x["center_x"])
            
            # Parse values
            quantity = 0
            rate = 0.0
            amount = 0.0
            
            if len(nearby_numerics) >= 3:
                quantity = self._parse_quantity(nearby_numerics[0]["text"])
                rate = self._parse_currency(nearby_numerics[1]["text"])
                amount = self._parse_currency(nearby_numerics[2]["text"])
            elif len(nearby_numerics) == 2:
                val1 = self._parse_currency(nearby_numerics[0]["text"])
                val2 = self._parse_currency(nearby_numerics[1]["text"])
                
                if val1 < 100 and val1 == int(val1):
                    quantity = int(val1)
                    amount = val2
                else:
                    amount = val1
            elif len(nearby_numerics) == 1:
                val = self._parse_currency(nearby_numerics[0]["text"])
                if val < 100 and val == int(val):
                    quantity = int(val)
                else:
                    amount = val
            
            # Create item
            items.append({
                "description": desc_pred["text"],
                "quantity": quantity,
                "rate": rate,
                "amount": amount,
            })
            
            print(f"   ✅ Pattern item: '{desc_pred['text'][:40]}...'")
            
            # Limit to 5 items
            if len(items) >= 5:
                break
        
        return items

    def _cluster_by_y_position(self, tokens: List[Dict], tolerance: float) -> Dict:
        """Cluster tokens by Y position (rows)"""
        clusters = {}
        
        for token in tokens:
            y_key = round(token["center_y"] / tolerance)
            if y_key not in clusters:
                clusters[y_key] = []
            clusters[y_key].append(token)
        
        return clusters

    def _cluster_by_x_position(self, tokens: List[Dict], tolerance: float) -> List[List[Dict]]:
        """Cluster tokens by X position (columns)"""
        if not tokens:
            return []
        
        sorted_tokens = sorted(tokens, key=lambda x: x["center_x"])
        
        columns = []
        current_col = [sorted_tokens[0]]
        
        for i in range(1, len(sorted_tokens)):
            token = sorted_tokens[i]
            prev_token = sorted_tokens[i-1]
            
            if abs(token["center_x"] - prev_token["center_x"]) < tolerance:
                current_col.append(token)
            else:
                columns.append(current_col)
                current_col = [token]
        
        if current_col:
            columns.append(current_col)
        
        return columns

    def _is_valid_item_description(self, description: str) -> bool:
        """Validate if text looks like a real product description"""
        if not description or len(description) < 5:
            return False
        
        text_lower = description.lower()
        
        # Common false positives to exclude
        false_positives = [
            "bill to", "ship to", "address", "date", "order", 
            "invoice", "total", "subtotal", "balance", "due",
            "payment", "terms", "thank", "page", "attention",
        ]
        
        for fp in false_positives:
            if fp in text_lower:
                return False
        
        # Check for product-like characteristics
        # 1. Has letters (not just numbers/symbols)
        if not any(c.isalpha() for c in description):
            return False
        
        # 2. Not just a number or currency
        if re.match(r'^[\$\€\£]?\s*\d+\.?\d*$', description):
            return False
        
        # 3. Not a date
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return False
        
        # 4. Good mix of characters for a product
        word_count = len(description.split())
        if word_count >= 2:  # Multi-word descriptions are more likely products
            return True
        
        # Single word but looks product-like (e.g., "Widget-X123")
        if re.search(r'[A-Za-z]+[-_][A-Za-z0-9]+', description):
            return True
        
        return True
    
    def _parse_quantity(self, text: str) -> int:
        """Parse quantity as integer"""
        match = re.search(r'[\d.]+', text)
        if match:
            try:
                return int(float(match.group(0)))
            except:
                pass
        return 0
    
    def _extract_invoice_number(self, predictions: List[Dict], img_w: int, img_h: int) -> str:
        """Extract invoice number - FIXED to prioritize # patterns"""
        patterns = [
            # Prioritize # patterns first
            r'#\s*([A-Z0-9\-]{3,})',
            r'INV[OICE]*[\s#:]*([A-Z0-9\-]{3,})',
            r'Invoice[\s#:]*([A-Z0-9\-]{3,})',
            r'No\.?\s*[:#]?\s*([A-Z0-9\-]{3,})',
            r'(\d{3,}[-_]\d+)',
        ]
        
        # Search in specific areas with priority
        search_areas = [
            # Area 1: Top-right (most common for invoice numbers)
            [p for p in predictions if p["center_y"] < img_h * 0.3 and p["center_x"] > img_w * 0.6],
            # Area 2: Anywhere with # symbol
            [p for p in predictions if '#' in p["text"]],
            # Area 3: All predictions (fallback)
            predictions
        ]
        
        for search_space in search_areas:
            for token in search_space:
                text = token["text"]
                
                # DEBUG
                # print(f"DEBUG invoice token: '{text}'")
                
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        invoice_num = match.group(1) if match.groups() else match.group(0)
                        
                        # Validate it's not something else
                        if invoice_num and len(invoice_num) >= 3:
                            # Skip if it looks like a date or other common false positives
                            if (not re.match(r'^\d{1,2}[/-]\d', invoice_num) and  # Not date
                                not invoice_num.upper() in ['ICE', 'DATE', 'DUE', 'TOTAL'] and  # Not labels
                                not self._looks_like_date(invoice_num)):
                                # print(f"DEBUG: Found invoice number: {invoice_num}")
                                return invoice_num
        
        # If not found, check vendor tokens for embedded invoice numbers
        vendor_area = [p for p in predictions if p["center_y"] < img_h * 0.2 and p["center_x"] < img_w * 0.4]
        for token in vendor_area:
            # Look for # pattern in vendor text
            match = re.search(r'#\s*(\d{3,})', token["text"])
            if match:
                return f"#{match.group(1)}"
        
        return ""
        
    def _extract_dates(self, predictions: List[Dict]) -> List[str]:
        """Extract date strings"""
        dates = []
        patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
        ]
        
        for pred in predictions:
            for pattern in patterns:
                if re.search(pattern, pred["text"], re.IGNORECASE):
                    dates.append(pred["text"])
                    break
        
        return dates[:2]
    
    def _parse_currency(self, text: str) -> float:
        """Parse currency value to float"""
        match = re.search(r'[\d,]+\.?\d*', text)
        if not match:
            return 0.0
        
        num_str = match.group(0)
        
        if ',' in num_str and '.' in num_str:
            if num_str.rfind(',') > num_str.rfind('.'):
                num_str = num_str.replace('.', '').replace(',', '.')
            else:
                num_str = num_str.replace(',', '')
        elif ',' in num_str:
            parts = num_str.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                num_str = num_str.replace(',', '.')
            else:
                num_str = num_str.replace(',', '')
        
        try:
            return float(num_str)
        except:
            return 0.0
    
    def _detect_currency(self, predictions: List[Dict]) -> str:
        """Detect currency from text"""
        for pred in predictions:
            text = pred["text"].upper()
            if '$' in text or 'USD' in text:
                return "USD"
            elif '€' in text or 'EUR' in text:
                return "EUR"
            elif '£' in text or 'GBP' in text:
                return "GBP"
            elif '¥' in text or 'JPY' in text:
                return "JPY"
            elif '₹' in text or 'INR' in text:
                return "INR"
        return "USD"
    
    def _visualize_predictions(self, image: Image.Image, predictions: List[Dict]) -> Image.Image:
        """Visualize predictions on image"""
        draw = ImageDraw.Draw(image)
        colors = {"HEADER": "red", "QUESTION": "blue", "ANSWER": "green", "OTHER": "orange"}
        
        for pred in predictions:
            if pred["label"] == "O":
                continue
            
            label_type = pred["label"].replace("B-", "").replace("I-", "")
            color = colors.get(label_type, "gray")
            
            box = pred["bbox"]
            draw.rectangle(box, outline=color, width=2)
            
            if pred.get("is_numeric"):
                draw.rectangle(box, outline="yellow", width=3)
            
            draw.text((box[0], max(0, box[1] - 15)), label_type[:6], fill=color)
        
        return image
    
    def _empty_result(self, image_path: str, width: int, height: int) -> Dict:
        """Return empty result structure"""
        return {
            "image_path": str(Path(image_path).resolve()),
            "image_size": {"width": width, "height": height},
            "raw_predictions": [],
            "entity_count": 0,
            "structured_data": {
                "vendor_company": "",
                "invoice_number": "",
                "invoice_date": "",
                "due_date": "",
                "order_id": "",
                "bill_to": "",
                "ship_to": "",
                "attention": "",
                "items": [],
                "subtotal": 0.0,
                "shipping": 0.0,
                "tax": 0.0,
                "discount": 0.0,
                "total_amount": 0.0,
                "balance_due": 0.0,
                "payment_terms": "",
                "notes": "",
                "currency": "USD",
                "extraction_confidence": 0.0,
                "extraction_timestamp": datetime.now().isoformat(),
            }
        }
    
    def save_results(self, result: Dict, output_dir: str = None):
        """Save results to files"""
        output_dir = Path(output_dir or "./invoice_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result["image_path"]).stem
        
        vis_path = output_dir / f"{stem}_visualized.png"
        result["visualization"].save(vis_path)
        
        struct_path = output_dir / f"{stem}_structured.json"
        with open(struct_path, 'w', encoding='utf-8') as f:
            json.dump(result["structured_data"], f, indent=2, ensure_ascii=False)
        
        items = result["structured_data"]["items"]
        if items:
            print(f"\n📦 LINE ITEMS ({len(items)}):")
            print(f"{'Description':<40} {'Qty':>6} {'Rate':>10} {'Amount':>10}")
            print("-" * 70)
            for item in items[:10]:
                desc = item["description"][:36] + "..." if len(item["description"]) > 36 else item["description"]
                qty = item["quantity"] if item["quantity"] else "-"
                rate = f"{item['rate']:.2f}" if item["rate"] else "-"
                amt = f"{item['amount']:.2f}"
                print(f"{desc:<40} {qty:>6} {rate:>10} {amt:>10}")
            if len(items) > 10:
                print(f"... and {len(items) - 10} more items")
        
        print(f"\n✅ Results saved to: {output_dir}")
        return output_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LiLT Invoice Parser - FIXED v5")
    parser.add_argument("image", help="Invoice image path")
    parser.add_argument("--model", default="./lilt_invoice_model", help="Model directory")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🧾 LiLT Invoice Parser - COMPREHENSIVE FIX v5")
    print("="*70)
    print("✅ FIXED: Vendor extraction (filters out address labels)")
    print("✅ FIXED: Address mixing (better boundary detection)")
    print("✅ FIXED: Shipping amount (stricter label verification)")
    print("✅ ADDED: Notes field extraction")
    print("✅ IMPROVED: Order ID detection (comprehensive patterns)")
    print("="*70)
    
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    try:
        parser = LiLTInvoiceParser(
            model_path=args.model,
            use_cuda=not args.no_cuda
        )
        
        result = parser.predict(args.image)
        parser.save_results(result, args.output)
        
        data = result["structured_data"]
        print("\n" + "="*70)
        print("📋 EXTRACTION RESULTS")
        print("="*70)
        
        categories = {
            "Vendor Info": ["vendor_company"],
            "Invoice Details": ["invoice_number", "invoice_date", "due_date", "order_id"],
            "Addresses": ["bill_to", "ship_to", "attention"],
            "Financials": ["subtotal", "shipping", "tax", "discount", "total_amount", "balance_due"],
            "Other": ["payment_terms", "notes", "currency"]
        }
        
        extracted = 0
        total = 0
        
        for category, fields in categories.items():
            print(f"\n{category}:")
            for field in fields:
                total += 1
                value = data.get(field)
                
                if field in ["bill_to", "ship_to"] and value and "\n" in value:
                    lines = value.split("\n")
                    print(f"  ✅ {field:15}: {len(lines)} lines")
                    extracted += 1
                elif field in ["subtotal", "shipping", "tax", "discount", "total_amount", "balance_due"]:
                    if isinstance(value, (int, float)) and value > 0:
                        print(f"  ✅ {field:15}: {data['currency']} {value:.2f}")
                        extracted += 1
                    else:
                        print(f"  ❌ {field:15}: Not found")
                elif field == "items" and value:
                    print(f"  ✅ {field:15}: {len(value)} items (see above)")
                    extracted += 1
                elif value and value not in ["", 0.0, 0, []]:
                    display = str(value)[:40] + "..." if len(str(value)) > 40 else str(value)
                    print(f"  ✅ {field:15}: {display}")
                    extracted += 1
                else:
                    print(f"  ❌ {field:15}: Not found")
        
        print(f"\n📊 Summary: {extracted}/{total} fields extracted | Confidence: {data.get('extraction_confidence', 0):.1%}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()