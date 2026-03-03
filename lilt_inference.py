#!/usr/bin/env python3
"""
lilt_inference_relation22.py – Extract HEADER/QUESTION/ANSWER from invoice (PDF/image)
+ Generates visualisation PNG.
+ Extracts relations between QUESTION and ANSWER tokens.
+ Outputs structured invoice JSON.

FIXES (r19):
- group_answer_tokens: numeric tokens NOT grouped with text tokens on same line
  (fixes "3 Strawberry" merging — "3" stays separate from "Strawberry").
- build_invoice_json Pass 4: description column uses a RANGE (left_x to right_x)
  instead of center±tol, so "Apple"/"Strawberry" are correctly captured even
  when they sit left of the DESCRIPTION header center.
- vendor_company: takes first QUESTION token at top-left of page (ABC Company Limited).
- invoice_date / due_date: Pass 1 now only uses tokens to the RIGHT of the
  question label (ax > qx), preventing left-side address tokens bleeding in.
- bill_to: stops collecting once it hits a line that contains a QUESTION token
  anywhere on the page at the same Y (prevents address spilling into table).
- order_id: correctly takes only PO# token (first a_tok to the right of PO.# label).
"""

import json
import re
import sys
import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import LayoutLMv3Processor, AutoConfig, AutoModelForTokenClassification
import easyocr
import logging

try:
    from pdf2image import convert_from_path
except ImportError:
    print("pdf2image not installed. Please install it: pip install pdf2image")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    model_dir: str = "lilt_model_11"
    token_labels: List[str] = field(default_factory=lambda: [
        "O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"
    ])
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    refine_labels: bool = True
    confidence_threshold: float = 0.0
    relation_confidence: float = 0.15
    max_pair_dist: int = 50
    spatial_dist: int = 600


# --------------------------------------------------------------------------
# OCR Helpers
# --------------------------------------------------------------------------
def _is_numeric_token(text: str) -> bool:
    clean = re.sub(r"[^\d.,]", "", text)
    return bool(re.match(r"^\d+\.?\d*$", clean)) and len(clean) >= 1


def _contains_currency(text: str) -> bool:
    return bool(re.search(r"[\$€£¥₹]|USD|EUR|GBP|HKD|HK\$", text))


def load_image(path: str) -> Optional[Image.Image]:
    path_lower = str(path).lower()
    if path_lower.endswith('.pdf'):
        try:
            pages = convert_from_path(path, first_page=1, last_page=1)
            if pages:
                return pages[0].convert("RGB")
            logger.error("PDF conversion returned no pages.")
            return None
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return None
    else:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            return None


def extract_ocr_tokens(image_path: str) -> Tuple[List[Dict], Tuple[int, int]]:
    image = load_image(image_path)
    if image is None:
        return [], (0, 0)
    w, h = image.size
    tokens = []

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        scale  = 2
        scaled = image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        results = reader.readtext(
            np.array(scaled), detail=1, paragraph=False,
            min_size=2, text_threshold=0.3, link_threshold=0.15,
            canvas_size=2560, low_text=0.15,
        )
        for bbox, text, conf in results:
            text = text.strip()
            if not text or conf < 0.05:
                continue
            x0 = int(min(p[0] for p in bbox) / scale)
            y0 = int(min(p[1] for p in bbox) / scale)
            x1 = int(max(p[0] for p in bbox) / scale)
            y1 = int(max(p[1] for p in bbox) / scale)
            is_single_digit = len(text) == 1 and text.isdigit()
            min_size = 1 if is_single_digit else 2
            if (x1 - x0) < min_size or (y1 - y0) < min_size:
                continue
            if text == "QTV":  text = "QTY"
            if text == "tOTal": text = "TOTAL"
            tokens.append({
                "text":      text,
                "bbox":      [x0, y0, x1, y1],
                "norm_bbox": [
                    max(0, min(1000, int(1000 * x0 / w))),
                    max(0, min(1000, int(1000 * y0 / h))),
                    max(0, min(1000, int(1000 * x1 / w))),
                    max(0, min(1000, int(1000 * y1 / h))),
                ],
                "center_x":  (x0 + x1) / 2,
                "center_y":  (y0 + y1) / 2,
                "confidence": conf,
                "is_numeric": _is_numeric_token(text),
                "is_currency": _contains_currency(text),
            })
        if tokens:
            tokens.sort(key=lambda t: (t["center_y"], t["center_x"]))
            logger.info(f"✅ EasyOCR: {len(tokens)} tokens")
            return tokens, (w, h)
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")

    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 6 --oem 1")
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            is_single_digit = len(text) == 1 and text.isdigit()
            min_conf = 30 if is_single_digit else 50
            if not text or conf < min_conf:
                continue
            x, y, wt, ht = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            min_dim = 1 if is_single_digit else 3
            if wt < min_dim or ht < min_dim:
                continue
            tokens.append({
                "text":      text,
                "bbox":      [x, y, x + wt, y + ht],
                "norm_bbox": [
                    max(0, min(1000, int(1000 * x / w))),
                    max(0, min(1000, int(1000 * y / h))),
                    max(0, min(1000, int(1000 * (x + wt) / w))),
                    max(0, min(1000, int(1000 * (y + ht) / h))),
                ],
                "center_x":  x + wt / 2,
                "center_y":  y + ht / 2,
                "confidence": conf / 100,
                "is_numeric": _is_numeric_token(text),
                "is_currency": _contains_currency(text),
            })
        if tokens:
            tokens.sort(key=lambda t: (t["center_y"], t["center_x"]))
            logger.info(f"✅ Tesseract: {len(tokens)} tokens")
            return tokens, (w, h)
    except Exception as e:
        logger.error(f"All OCR engines failed: {e}")

    return [], (w, h)


# --------------------------------------------------------------------------
# Label refinement
# --------------------------------------------------------------------------
def refine_labels(predictions: List[Dict], img_w: int, img_h: int) -> List[Dict]:
    def base(lbl: str) -> str:
        return lbl[2:] if isinstance(lbl, str) and lbl.startswith(('B-', 'I-')) else lbl

    line_tol = img_h * 0.02
    wide_tol = img_h * 0.05

    for t in predictions:
        if t.get('center_x') is None:
            b = t.get('bbox', [0, 0, 0, 0])
            t['center_x'] = (b[0] + b[2]) / 2
            t['center_y'] = (b[1] + b[3]) / 2
        if 'is_numeric' not in t:
            t['is_numeric'] = _is_numeric_token(t.get('text', ''))

    col_hdr_kws = {"qty", "quantity", "description", "unit price", "amount", "rate", "price",
                   "item", "product", "code", "part", "sku"}

    header_candidates = []
    for p in predictions:
        tl = p['text'].lower().strip()
        if any(k in tl for k in col_hdr_kws):
            if img_h * 0.1 < p['center_y'] < img_h * 0.7:
                header_candidates.append(p['center_y'])

    header_row_y = float(np.median(header_candidates)) if header_candidates else None
    table_top    = (header_row_y + img_h * 0.02) if header_row_y else img_h * 0.35
    table_bottom = img_h * 0.75

    for p in predictions:
        tl = p['text'].lower().strip()
        if tl in ('total', 'subtotal') and p['center_y'] > img_h * 0.4:
            table_bottom = min(table_bottom, p['center_y'] + img_h * 0.01)
            break

    logger.info(f"📊 Column header row y={header_row_y}  table: y={table_top:.0f}–{table_bottom:.0f}")

    data_rows_y = set()
    for p in predictions:
        if base(p['label']) == 'ANSWER' and not p['is_numeric']:
            data_rows_y.add(p['center_y'])

    for t in predictions:
        if not t['is_numeric'] or base(t['label']) == 'ANSWER':
            continue
        cy = t['center_y']
        if not (table_top < cy < table_bottom):
            continue
        if any(abs(cy - dr) < line_tol for dr in data_rows_y):
            t['label'] = 'ANSWER'

    qty_tok = next((p for p in predictions
                    if p['text'].lower().strip() in ('qty', 'quantity', 'qty.')), None)
    if qty_tok:
        qty_x    = qty_tok['center_x']
        col_band = img_w * 0.09
        for t in predictions:
            if not t['is_numeric'] or base(t['label']) == 'ANSWER':
                continue
            if abs(t['center_x'] - qty_x) > col_band or t['center_y'] <= qty_tok['center_y']:
                continue
            if (table_top < t['center_y'] < table_bottom
                    or any(abs(t['center_y'] - dr) < wide_tol for dr in data_rows_y)):
                old = t['label']
                t['label'] = 'ANSWER'
                logger.info(f"  QTY-band → ANSWER: '{t['text']}' (was {old})")

    desc_answers = [p for p in predictions
                    if base(p['label']) == 'ANSWER' and not p['is_numeric']
                    and table_top < p['center_y'] < table_bottom]
    for desc in desc_answers:
        for t in predictions:
            if not t['is_numeric'] or base(t['label']) == 'ANSWER':
                continue
            if t['center_x'] >= desc['center_x'] or abs(t['center_y'] - desc['center_y']) > wide_tol:
                continue
            if header_row_y and t['center_y'] <= header_row_y:
                continue
            old = t['label']
            t['label'] = 'ANSWER'
            logger.info(f"  desc-neighbour → ANSWER: '{t['text']}' (was {old})")

    if header_row_y:
        for t in predictions:
            if abs(t['center_y'] - header_row_y) < line_tol:
                tl = t['text'].lower().strip()
                if any(k in tl for k in col_hdr_kws) and base(t['label']) == 'ANSWER':
                    t['label'] = 'QUESTION'
                    logger.info(f"  col-header → QUESTION: '{t['text']}'")

    field_label_kws = {"invoice #", "invoice date", "p.o.#", "due date", "bill to", "ship to",
                       "po#", "p.o.", "order", "terms"}
    for t in predictions:
        tl = t['text'].lower().strip()
        if any(k in tl for k in field_label_kws) and base(t['label']) == 'ANSWER':
            t['label'] = 'QUESTION'
            logger.info(f"  field-label → QUESTION: '{t['text']}'")

    answer_value_pats = [r'^\d{4}/\d{2}/\d{2}$', r'^\d{4}-\d{2}-\d{2}$',
                         r'^[A-Z]{2,4}-\d{4,}', r'^PO-']
    for t in predictions:
        if base(t['label']) not in ('QUESTION', 'O'):
            continue
        if any(re.match(pat, t['text'].strip()) for pat in answer_value_pats):
            t['label'] = 'ANSWER'
            logger.info(f"  value-pattern → ANSWER: '{t['text']}'")

    terms_kws = {"terms & conditions", "terms and conditions"}
    term_hdr = next((p for p in sorted(predictions, key=lambda x: x['center_y'])
                     if any(k in p['text'].lower() for k in terms_kws)
                     and p['center_y'] > img_h * 0.6), None)
    if term_hdr:
        for p in predictions:
            if p is term_hdr:
                continue
            if term_hdr['center_y'] - line_tol < p['center_y'] < term_hdr['center_y'] + img_h * 0.25:
                if base(p['label']) == 'QUESTION' and not p['is_numeric']:
                    p['label'] = 'ANSWER'

    pay_pats = [r'payment\s+is\s+due\s+within\s+\d+\s+days', r'net\s+\d+', r'due\s+(?:upon|on)\s+receipt']
    line_groups: Dict[float, List] = {}
    for p in predictions:
        cy    = p['center_y']
        found = False
        for ly in line_groups:
            if abs(cy - ly) < line_tol:
                line_groups[ly].append(p)
                found = True
                break
        if not found:
            line_groups[cy] = [p]
    for ly, toks in line_groups.items():
        lt = ' '.join(t['text'] for t in sorted(toks, key=lambda x: x['center_x']))
        if any(re.search(pat, lt, re.I) for pat in pay_pats) and ':' not in lt:
            for t in toks:
                if base(t['label']) == 'QUESTION':
                    t['label'] = 'ANSWER'

    decorative = {"thank you", "powered by", "thank", "regards", "sincerely"}
    for t in predictions:
        tl = t['text'].lower().strip()
        if any(d in tl for d in decorative) and base(t['label']) == 'ANSWER':
            t['label'] = 'O'
            logger.info(f"  decorative → O: '{t['text']}'")

    return predictions


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def visualize(image: Image.Image, predictions: List[Dict]) -> Image.Image:
    draw    = ImageDraw.Draw(image)
    cmap    = {'HEADER': 'red', 'QUESTION': 'blue', 'ANSWER': 'limegreen', 'O': 'gray'}
    colored = 0
    for p in predictions:
        lbl   = p.get('label', 'O')
        color = cmap.get(lbl, 'gray')
        if color == 'gray':
            continue
        box = p.get('bbox', [0, 0, 0, 0])
        if box[2] > box[0] and box[3] > box[1]:
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], max(0, box[1] - 15)), lbl[:4].upper(),
                      fill=color, stroke_width=1, stroke_fill='white')
            colored += 1
    logger.info(f"🎨 Colored {colored} tokens")
    return image


# --------------------------------------------------------------------------
# Group answer tokens — numeric and text tokens kept SEPARATE
# --------------------------------------------------------------------------
def group_answer_tokens(tokens: List[Dict], img_w: int, img_h: int,
                        line_tol_factor: float = 0.02,
                        word_gap_factor: float = 0.07) -> List[Dict]:
    """
    Group adjacent ANSWER tokens on the same line.

    KEY FIX (r19): numeric tokens and text tokens are NEVER merged into the
    same group. This prevents "3" (QTY) from joining "Strawberry" (DESCRIPTION)
    even when they happen to be close horizontally.

    Also: group width capped at 40% of page to prevent cross-column merging.
    """
    line_tol  = img_h * line_tol_factor
    word_gap  = img_w * word_gap_factor
    max_width = img_w * 0.40

    answer_tokens = [t for t in tokens if t['label'] == 'ANSWER']
    if not answer_tokens:
        return []

    # Group by horizontal line
    lines: Dict[float, List] = {}
    for t in answer_tokens:
        matched = False
        for y in list(lines.keys()):
            if abs(t['center_y'] - y) < line_tol:
                lines[y].append(t)
                matched = True
                break
        if not matched:
            lines[t['center_y']] = [t]

    grouped = []
    for y, line in lines.items():
        line.sort(key=lambda t: t['center_x'])
        current = [line[0]]
        for t in line[1:]:
            prev         = current[-1]
            gap          = t['bbox'][0] - prev['bbox'][2]
            group_width  = t['bbox'][2] - current[0]['bbox'][0]
            # FIX: never merge numeric with non-numeric (or vice versa)
            type_mismatch = (t.get('is_numeric', False) != prev.get('is_numeric', False))
            if gap < word_gap and group_width < max_width and not type_mismatch:
                current.append(t)
            else:
                grouped.append(current)
                current = [t]
        grouped.append(current)

    result = []
    for group in grouped:
        result.append({
            'text':       ' '.join(t['text'] for t in group).strip(),
            'bbox':       [min(t['bbox'][0] for t in group),
                           min(t['bbox'][1] for t in group),
                           max(t['bbox'][2] for t in group),
                           max(t['bbox'][3] for t in group)],
            'center_x':   float(np.mean([t['center_x'] for t in group])),
            'center_y':   float(np.mean([t['center_y'] for t in group])),
            'confidence': float(np.mean([t['confidence'] for t in group])),
            'label':      'ANSWER',
            'is_numeric': group[0].get('is_numeric', False),
            'group':      [t['text'] for t in group],
        })
    return result


# --------------------------------------------------------------------------
# Relation Extraction
# --------------------------------------------------------------------------
def extract_relations(tokens: List[Dict], config: InferenceConfig,
                      img_w: int, img_h: int) -> List[Dict]:
    same_line_tol = img_h * 0.025
    col_align_tol = img_w * 0.06

    KNOWN_FIELD_LABELS = {
        "invoice #", "invoice date", "due date", "p.o.#", "po.#", "po#",
        "p.o. #", "bill to", "ship to", "payment terms", "terms & conditions",
        "balance due", "total", "subtotal", "tax", "discount", "shipping",
        "attention", "currency", "order #", "date", "ref #", "reference",
    }
    TABLE_COL_LABELS = {"qty", "quantity", "description", "item",
                        "unit price", "price", "rate", "amount"}

    def is_real_question(q: Dict) -> bool:
        qt = q['text'].lower().strip().rstrip(':').strip()
        if qt in KNOWN_FIELD_LABELS:
            return True
        if any(kw in qt for kw in TABLE_COL_LABELS):
            return True
        if len(qt.split()) > 2:
            return False
        cx_frac = q['center_x'] / img_w
        if 0.15 < cx_frac < 0.85 and qt not in KNOWN_FIELD_LABELS:
            if not re.search(r'[#:@]', qt):
                ABBREV_LABELS = {"qty", "ref", "attn", "po", "vat", "gst",
                                 "hsn", "sku", "upc", "item", "date", "no",
                                 "terms", "amount", "price", "rate", "desc"}
                if qt not in ABBREV_LABELS:
                    return False
        return True

    def q_semantic_key(q_text: str) -> str:
        qt = q_text.lower().strip().rstrip(':').strip()
        if re.search(r"inv[o0]ice\s*[#n]", qt):             return "invoice_number"
        if re.search(r"inv[o0]ice\s*date", qt):              return "invoice_date"
        if re.search(r"due\s*date", qt):                      return "due_date"
        if re.search(r"p\.?o\.?\s*#|purchase\s*order", qt):  return "po_number"
        if re.search(r"order\s*#", qt):                       return "order_number"
        if re.search(r"bill\s*to", qt):                       return "bill_to"
        if re.search(r"ship\s*to", qt):                       return "ship_to"
        if re.search(r"payment\s*term", qt):                  return "payment_terms"
        if re.search(r"terms\s*&?\s*conditions?", qt):        return "terms"
        if re.search(r"balance\s*due", qt):                   return "balance_due"
        if re.search(r"\btotal\b", qt):                       return "total"
        if re.search(r"subtotal", qt):                        return "subtotal"
        if re.search(r"\btax\b", qt):                         return "tax"
        if re.search(r"discount", qt):                        return "discount"
        if re.search(r"shipping|freight", qt):                return "shipping"
        if re.search(r"\bqty\b|\bquantity\b", qt):           return "qty_col"
        if re.search(r"desc|item|product", qt):               return "desc_col"
        if re.search(r"unit\s*price|rate", qt):               return "unit_price_col"
        if re.search(r"\bamount\b", qt):                      return "amount_col"
        return qt

    questions     = [t for t in tokens if t['label'] == 'QUESTION' and is_real_question(t)]
    answer_groups = group_answer_tokens(tokens, img_w, img_h)

    if not questions or not answer_groups:
        return []

    # ── Build table column headers with X-range boundaries (same as build_invoice_json) ──
    table_col_qs = [q for q in questions if any(kw in q['text'].lower() for kw in TABLE_COL_LABELS)]
    header_row_y = float(np.median([q['center_y'] for q in table_col_qs])) if table_col_qs else None

    # Build col_headers with left/right range so DESCRIPTION captures answers
    # that sit LEFT of the column header center (e.g. "Apple" at x=320, header at x=562)
    # Build column ranges keyed by column NAME (not object identity)
    col_headers_rel: List[Dict] = []
    if table_col_qs and header_row_y is not None:
        qs_sorted = sorted(table_col_qs, key=lambda q: q['center_x'])
        for i, q in enumerate(qs_sorted):
            left_x  = (qs_sorted[i-1]['center_x'] + q['center_x']) / 2 if i > 0 else 0
            right_x = (q['center_x'] + qs_sorted[i+1]['center_x']) / 2 if i < len(qs_sorted)-1 else img_w
            col_headers_rel.append({
                'name':     q['text'].lower().strip(),
                'center_x': q['center_x'],
                'left_x':   left_x,
                'right_x':  right_x,
            })
        logger.info(f"  Relation col ranges: {[(c['name'], round(c['left_x']), round(c['right_x'])) for c in col_headers_rel]}")

    def get_col_name_for_x(ax: float) -> Optional[str]:
        for col in col_headers_rel:
            if col['left_x'] <= ax <= col['right_x']:
                return col['name']
        return None

    def q_col_name(q: Dict) -> Optional[str]:
        for col in col_headers_rel:
            if col['left_x'] <= q['center_x'] <= col['right_x']:
                return col['name']
        return None

    logger.info(f"  Filtered questions: {[q['text'] for q in questions]}")
    logger.info(f"  Table header row y={header_row_y}")
    logger.info(f"  Answer groups: {[a['text'] for a in answer_groups]}")

    best_for_q: Dict[str, Dict] = {}
    max_dist = float(config.spatial_dist)

    for q in questions:
        qx, qy  = q['center_x'], q['center_y']
        q_conf  = max(q.get('confidence', 0.5), 0.5)
        q_key   = q_semantic_key(q['text'])
        is_col  = header_row_y is not None and abs(qy - header_row_y) < same_line_tol * 2
        this_col_name = q_col_name(q) if is_col else None

        for a in answer_groups:
            ax, ay = a['center_x'], a['center_y']

            if is_col:
                if ay <= header_row_y + same_line_tol:
                    continue
                # FIX: compare column names (strings), not object identity
                answer_col_name = get_col_name_for_x(ax)
                if answer_col_name is None or answer_col_name != this_col_name:
                    continue
                dy    = ay - header_row_y
                # FIX: use average (not product) of confidences to avoid near-zero scores
                score = ((q_conf + a.get('confidence', 0.5)) / 2.0) * (1.0 - min(dy, max_dist) / max_dist)
                if score < config.relation_confidence:
                    continue
                row_bucket = round(ay / (img_h * 0.04))
                key = f"{q_key}_row{row_bucket}"
            else:
                if ax < qx - img_w * 0.05:
                    continue
                if ay > qy + img_h * 0.08:
                    continue
                if ay < qy - same_line_tol:
                    continue
                dx            = abs(qx - ax)
                dy            = abs(qy - ay)
                weighted_dist = dx + dy * 3.0
                if weighted_dist > max_dist:
                    continue
                same_line  = dy < same_line_tol
                line_bonus = 0.3 if same_line else 0.0
                proximity  = 1.0 - (weighted_dist / max_dist)
                score      = ((q_conf + a.get('confidence', 0.5)) / 2.0) * proximity + line_bonus
                if score < config.relation_confidence:
                    continue
                key = q_key

            cand = {
                'question':      q['text'],
                'answer':        a['text'],
                'confidence':    round(float(score), 4),
                'same_line':     int(abs(ay - qy) < same_line_tol),
                'question_bbox': q.get('bbox'),
                'answer_bbox':   a.get('bbox'),
            }
            if key not in best_for_q or cand['confidence'] > best_for_q[key]['confidence']:
                best_for_q[key] = cand

    # BILL TO: full multi-line address — collect ANSWER tokens below BILL TO
    bill_to_q = next((q for q in questions if re.search(r"bill\s*to", q['text'], re.I)), None)
    if bill_to_q:
        bqy    = bill_to_q['center_y']
        # Stop before the table header row
        max_ay = (header_row_y - img_h * 0.02) if header_row_y else bqy + img_h * 0.20
        addr_lines = []
        for a in sorted(answer_groups, key=lambda a: a['center_y']):
            if a['center_y'] <= bqy + same_line_tol:
                continue
            if a['center_y'] > max_ay:
                break
            # Left half only, no pure numerics (table QTY/amounts leak in otherwise)
            if a['center_x'] < img_w * 0.45 and not a.get('is_numeric', False):
                addr_lines.append(('answer', a['center_y'], a['text']))
        # Sort and deduplicate
        addr_lines.sort(key=lambda x: x[1])
        seen_texts = set()
        deduped = []
        for _, _, txt in addr_lines:
            if txt not in seen_texts:
                seen_texts.add(txt)
                deduped.append(txt)
        if deduped:
            best_for_q['bill_to'] = {
                'question':      'BILL TO',
                'answer':        '\n'.join(deduped),
                'confidence':    1.0,
                'same_line':     0,
                'question_bbox': bill_to_q.get('bbox'),
                'answer_bbox':   [0, 0, 0, 0],
            }
            logger.info(f"  BILL TO → {deduped}")

    # TOTAL: ANSWER-labeled TOTAL token → value to its right
    total_tok = next((t for t in tokens
                      if t['label'] == 'ANSWER'
                      and t['text'].upper() in ('TOTAL', 'TOTAL:')), None)
    if total_tok:
        tx, ty     = total_tok['center_x'], total_tok['center_y']
        candidates = [a for a in answer_groups
                      if abs(a['center_y'] - ty) < same_line_tol
                      and a['center_x'] > tx
                      and a['text'].upper() not in ('TOTAL', 'TOTAL:')]
        if candidates:
            best_total = min(candidates, key=lambda a: a['center_x'])
            best_for_q['total'] = {
                'question':      'TOTAL',
                'answer':        best_total['text'],
                'confidence':    1.0,
                'same_line':     1,
                'question_bbox': total_tok.get('bbox'),
                'answer_bbox':   best_total.get('bbox'),
            }
            logger.info(f"  TOTAL → '{best_total['text']}'")

    relations = sorted(best_for_q.values(), key=lambda r: r['confidence'], reverse=True)
    logger.info(f"🔗 {len(relations)} relations extracted")
    return relations


# --------------------------------------------------------------------------
# Label extraction class
# --------------------------------------------------------------------------
class LabelExtractor:
    def __init__(self, config: InferenceConfig):
        self.config    = config
        self.processor = None
        self.model     = None
        self.id2label  = {}
        self._load_model()

    def _load_model(self):
        mp = Path(self.config.model_dir)
        lm = mp / "label_map.json"
        if lm.exists():
            with open(lm) as f:
                self.id2label = {int(k): v for k, v in json.load(f).items()}
        else:
            self.id2label = {i: l for i, l in enumerate(self.config.token_labels)}
            logger.warning("label_map.json not found, using defaults")

        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False)

        cfg = AutoConfig.from_pretrained(mp, ignore_mismatched_sizes=True)
        cfg.num_labels = len(self.id2label)
        self.model = AutoModelForTokenClassification.from_pretrained(
            mp, config=cfg, ignore_mismatched_sizes=True)
        self.model.to(self.config.device).eval()
        logger.info(f"✅ Model loaded from {self.config.model_dir}")

    def extract(self, image_path: str) -> Dict:
        tokens, (img_w, img_h) = extract_ocr_tokens(image_path)
        if not tokens:
            logger.error("No tokens extracted")
            return {"error": "No tokens"}

        words     = [t["text"] for t in tokens]
        boxes     = [t["norm_bbox"] for t in tokens]
        dummy_img = Image.new("RGB", (224, 224), color="white")

        encoding = self.processor(
            images=dummy_img, text=words, boxes=boxes,
            padding="max_length", truncation=True,
            max_length=self.config.max_length, return_tensors="pt",
        )
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(self.config.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                bbox=encoding["bbox"],
                attention_mask=encoding["attention_mask"],
            )
        probs    = torch.softmax(outputs.logits, dim=-1)[0]
        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        confs    = torch.max(probs, dim=-1).values.cpu().numpy().tolist()

        word_ids = encoding.word_ids(0)
        aligned  = []
        seen     = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= len(tokens):
                continue
            seen.add(wid)
            tok        = tokens[wid].copy()
            raw_label  = self.id2label.get(int(pred_ids[idx]), "O")
            base_label = raw_label[2:] if raw_label.startswith(("B-", "I-")) else raw_label
            confidence = confs[idx]
            if self.config.confidence_threshold > 0 and confidence < self.config.confidence_threshold:
                base_label = "O"
            tok["model_label"] = raw_label
            tok["label"]       = base_label
            tok["confidence"]  = round(float(confidence), 4)
            aligned.append(tok)

        if self.config.refine_labels:
            aligned = refine_labels(aligned, img_w, img_h)

        counts = {"HEADER": 0, "QUESTION": 0, "ANSWER": 0, "O": 0}
        for t in aligned:
            counts[t["label"]] += 1

        orig_image = load_image(image_path)
        vis_image  = visualize(orig_image.copy(), aligned) if orig_image else None

        return {
            "image_path":    str(Path(image_path).resolve()),
            "image_size":    {"width": img_w, "height": img_h},
            "tokens":        aligned,
            "stats":         counts,
            "visualization": vis_image,
        }


# --------------------------------------------------------------------------
# Structured Invoice Builder
# --------------------------------------------------------------------------
def build_invoice_json(result: Dict, image_path: str) -> Dict:
    tokens   = result["tokens"]
    img_w    = result["image_size"]["width"]
    img_h    = result["image_size"]["height"]
    line_tol = img_h * 0.02

    # ── Group tokens into sorted lines ───────────────────────────────────────
    lines: Dict[float, List] = {}
    for t in tokens:
        matched = False
        for y in list(lines.keys()):
            if abs(t["center_y"] - y) < line_tol:
                lines[y].append(t)
                matched = True
                break
        if not matched:
            lines[t["center_y"]] = [t]

    sorted_lines = sorted(lines.items(), key=lambda kv: kv[0])
    for _, lt in sorted_lines:
        lt.sort(key=lambda t: t["center_x"])

    def parse_float(text: str) -> float:
        cleaned = re.sub(r"[^\d.]", "", text)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def detect_currency(toks: List[Dict]) -> str:
        for t in toks:
            txt = t.get("text", "")
            if re.search(r"HK\$|HKD|HKS", txt, re.I): return "HKD"
            if re.search(r"\$|USD", txt):               return "USD"
            if re.search(r"€|EUR", txt):                return "EUR"
            if re.search(r"£|GBP", txt):                return "GBP"
        return "USD"

    # ── Invoice skeleton ──────────────────────────────────────────────────────
    invoice = {
        "vendor_company":        "",
        "invoice_number":        "",
        "invoice_date":          "",
        "due_date":              "",
        "order_id":              "",
        "bill_to":               "",
        "ship_to":               "",
        "attention":             "",
        "items":                 [],
        "subtotal":              0.0,
        "shipping":              0.0,
        "tax":                   0.0,
        "discount":              0.0,
        "total_amount":          0.0,
        "balance_due":           0.0,
        "payment_terms":         "",
        "notes":                 "",
        "currency":              detect_currency(tokens),
        "extraction_confidence": round(float(np.mean([t.get("confidence", 0) for t in tokens])), 4),
        "extraction_timestamp":  datetime.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    FIELD_MAP = {
        r"invoice\s*#":      "invoice_number",
        r"invoice\s*date":   "invoice_date",
        r"due\s*date":       "due_date",
        r"p\.?o\.?\s*#":    "order_id",
        r"bill\s*to":        "bill_to",
        r"ship\s*to":        "ship_to",
        r"attention":        "attention",
        r"payment\s*terms":  "payment_terms",
        r"notes":            "notes",
        r"subtotal":         "subtotal",
        r"shipping|freight": "shipping",
        r"\btax\b":          "tax",
        r"discount":         "discount",
        r"balance\s*due":    "balance_due",
        r"\btotal\b":        "total_amount",
    }

    TABLE_COL_KEYWORDS = {"qty", "quantity", "description", "item",
                          "unit price", "price", "rate", "amount"}

    # ── Detect table header row — search ANY label, not just QUESTION ─────────
    # (QTY/AMOUNT may be labeled ANSWER by the model; we detect by text content)
    header_line = None
    header_y    = None
    for y, line in sorted_lines:
        # Count tokens whose text matches a column keyword (any label)
        col_kw_count = sum(
            any(k in t["text"].lower() for k in TABLE_COL_KEYWORDS)
            for t in line
        )
        if col_kw_count >= 2:
            header_line = line
            header_y    = y
            # Force all column-keyword tokens on this line to QUESTION
            for t in header_line:
                if any(k in t["text"].lower() for k in TABLE_COL_KEYWORDS):
                    t["label"] = "QUESTION"
            break

    # Build non-overlapping column boundaries sorted left-to-right
    col_headers: List[Dict] = []
    if header_line:
        # Use all tokens whose text matches a column keyword (labels already fixed above)
        qs = [t for t in header_line
              if any(k in t["text"].lower() for k in TABLE_COL_KEYWORDS)]
        qs.sort(key=lambda t: t["center_x"])
        for i, t in enumerate(qs):
            left_x  = (qs[i-1]["center_x"] + t["center_x"]) / 2 if i > 0 else 0
            right_x = (t["center_x"] + qs[i+1]["center_x"]) / 2 if i < len(qs)-1 else img_w
            col_headers.append({
                "name":     t["text"].lower().strip(),
                "center_x": t["center_x"],
                "left_x":   left_x,
                "right_x":  right_x,
            })
        # CRITICAL FIX: For description-like columns, extend left boundary to 0
        # because description text is left-aligned under QTY header on many invoices
        # (e.g. "Apple" at x=320 sits left of the DESCRIPTION header center at x=562)
        for col in col_headers:
            if any(k in col["name"] for k in ("desc", "item", "product")):
                col["left_x"] = 0
                logger.info(f"  Extended '{col['name']}' left boundary to 0 (left-aligned descriptions)")
        logger.info(f"  Table columns (non-overlapping): {[(c['name'], round(c['left_x']), round(c['right_x'])) for c in col_headers]}")
        logger.info(f"  header_y={header_y:.0f}")

    # ── Compute table_bottom from TOTAL token (sorted by Y) ──────────────────
    total_answer_tok = next(
        (t for t in sorted(tokens, key=lambda t: t["center_y"])
         if t["label"] == "ANSWER" and t["text"].upper() in ("TOTAL", "TOTAL:")),
        None
    )
    table_bottom = (total_answer_tok["center_y"] - line_tol
                    if total_answer_tok else img_h * 0.80)
    logger.info(f"  Table bottom y={table_bottom:.0f}")

    # ── Pass 1: key-value pairs — answer must be RIGHT of question ────────────
    for y, line in sorted_lines:
        q_toks = [t for t in line if t["label"] == "QUESTION"]
        if not q_toks:
            continue
        q_text = " ".join(t["text"] for t in q_toks).lower().strip()
        # Only ANSWER tokens to the RIGHT of the rightmost question token
        q_right_edge = max(t["bbox"][2] for t in q_toks)
        a_toks = [t for t in line
                  if t["label"] == "ANSWER" and t["center_x"] > q_right_edge]
        if not a_toks:
            continue
        a_text = " ".join(t["text"] for t in a_toks).strip()
        for pattern, f_field in FIELD_MAP.items():
            if re.search(pattern, q_text, re.I):
                if f_field in ("subtotal", "shipping", "tax", "discount",
                               "balance_due", "total_amount"):
                    invoice[f_field] = parse_float(a_text)
                elif f_field == "order_id":
                    # First token to the right of PO# label only
                    invoice[f_field] = a_toks[0]["text"].strip()
                else:
                    invoice[f_field] = a_text
                break

    # ── Pass 2: BILL TO — multi-line address, left half, stop at table ────────
    bill_q = next(
        (t for t in tokens
         if t["label"] == "QUESTION" and re.search(r"bill\s*to", t["text"], re.I)),
        None
    )
    if bill_q:
        bqy    = bill_q["center_y"]
        # Hard stop: just before table header row
        max_ay = (header_y - line_tol) if header_y else bqy + img_h * 0.25

        # Known field-label texts to exclude (right-side labels like INVOICE #, DUE DATE)
        FIELD_LABEL_KWS = {"invoice", "due date", "p.o.", "po#", "ship to", "bill to",
                           "qty", "quantity", "description", "unit price", "amount",
                           "subtotal", "total", "tax", "discount", "shipping", "terms"}

        addr_lines = []
        for y, line in sorted_lines:
            if y <= bqy + line_tol:
                continue
            if y >= max_ay:
                break
            # Collect all non-numeric, non-field-label tokens in the left half
            # regardless of label (ANSWER, O, or mislabeled QUESTION)
            a_toks = []
            for t in line:
                if t["center_x"] >= img_w * 0.50:
                    continue
                if t.get("is_numeric", False):
                    continue
                tl = t["text"].lower().strip()
                if any(kw in tl for kw in FIELD_LABEL_KWS):
                    continue
                if t["label"] in ("ANSWER", "O", "QUESTION"):
                    a_toks.append(t)
            if a_toks:
                line_text = " ".join(t["text"] for t in sorted(a_toks, key=lambda t: t["center_x"])).strip()
                if line_text:
                    addr_lines.append(line_text)

        if addr_lines:
            invoice["bill_to"] = "\n".join(addr_lines)
        logger.info(f"  BILL TO → {invoice['bill_to']!r}")

    # ── Pass 3: Vendor name — first QUESTION at top-left of page ─────────────
    # "ABC Company Limited" is labeled QUESTION at top-left; INVOICE is HEADER
    top_left_qs = [t for t in tokens
                   if t["label"] == "QUESTION"
                   and t["center_y"] < img_h * 0.20
                   and t["center_x"] < img_w * 0.50]
    if top_left_qs:
        invoice["vendor_company"] = sorted(top_left_qs, key=lambda t: t["center_y"])[0]["text"]

    if not invoice["vendor_company"]:
        header_toks = [t for t in tokens if t["label"] == "HEADER"
                       and t["text"].upper() not in ("INVOICE", "INVOICE:")]
        if header_toks:
            invoice["vendor_company"] = sorted(header_toks, key=lambda t: t["center_y"])[0]["text"]

    # ── Pass 4: Line items using column X-range matching ─────────────────────
    if header_y and col_headers:

        def get_column(cx: float, is_numeric: bool = False) -> Optional[Dict]:
            """Return the best column for this x position.
            When description extends left to 0, numeric tokens prefer qty/amount,
            text tokens prefer description."""
            candidates = [col for col in col_headers if col["left_x"] <= cx <= col["right_x"]]
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            # Numeric tokens → prefer qty/amount columns
            if is_numeric:
                num_cols = [c for c in candidates if any(k in c["name"] for k in ("qty", "quantity", "amount", "price", "rate"))]
                if num_cols:
                    return min(num_cols, key=lambda c: abs(c["center_x"] - cx))
            # Text tokens → prefer description column
            desc_cols = [c for c in candidates if any(k in c["name"] for k in ("desc", "item", "product"))]
            if desc_cols:
                return desc_cols[0]
            return min(candidates, key=lambda c: abs(c["center_x"] - cx))

        TABLE_HEADER_TEXTS = {"qty", "quantity", "description", "item",
                              "unit price", "price", "rate", "amount"}

        def is_table_token(t: Dict) -> bool:
            if t["label"] == "ANSWER":
                return True
            # O-labeled numerics aligned with a column (e.g. QTY "1" labeled O)
            if t["label"] == "O" and t.get("is_numeric"):
                return get_column(t["center_x"], t.get("is_numeric", False)) is not None
            # QUESTION-labeled non-header tokens inside table (e.g. "Bag" mislabeled QUESTION)
            if t["label"] == "QUESTION":
                if t["text"].lower().strip() in TABLE_HEADER_TEXTS:
                    return False   # skip actual column headers
                col = get_column(t["center_x"], t.get("is_numeric", False))
                if col and any(k in col["name"] for k in ("desc", "item", "product")):
                    return True   # mislabeled description word
            return False

        data_rows: Dict[float, List] = {}
        for y, line in sorted_lines:
            if y <= header_y + line_tol:
                continue
            if y > table_bottom:
                continue
            line_str = " ".join(t["text"] for t in line).upper()
            if "TOTAL" in line_str:
                continue
            if any(re.search(pat, line_str, re.I) for pat in
                   [r"payment\s+is\s+due", r"thank\s+you", r"powered\s+by"]):
                continue
            eligible = [t for t in line if is_table_token(t)]
            if eligible:
                data_rows[y] = eligible

        logger.info(f"  Data rows: { {round(y): [t['text'] for t in toks] for y, toks in data_rows.items()} }")

        for row_y, a_toks in sorted(data_rows.items()):
            row: Dict = {"description": "", "quantity": 0.0, "unit_price": 0.0, "amount": 0.0}

            for t in a_toks:
                col = get_column(t["center_x"], t.get("is_numeric", False))
                if col is None:
                    continue
                col_name = col["name"]
                text     = t["text"].strip()

                if any(k in col_name for k in ("desc", "item", "product")):
                    row["description"] = (row["description"] + " " + text).strip()
                elif any(k in col_name for k in ("qty", "quantity")):
                    try:
                        row["quantity"] = float(text.replace(",", ""))
                    except ValueError:
                        pass
                elif any(k in col_name for k in ("unit price", "price", "rate")):
                    row["unit_price"] = parse_float(text)
                elif "amount" in col_name:
                    row["amount"] = parse_float(text)

            if row["description"] or row["amount"] != 0.0:
                invoice["items"].append(row)

        logger.info(f"  Extracted {len(invoice['items'])} items: {invoice['items']}")

    # ── Pass 5: TOTAL from ANSWER-labeled TOTAL token ─────────────────────────
    if total_answer_tok:
        tx, ty  = total_answer_tok["center_x"], total_answer_tok["center_y"]
        same_ln = [t for t in tokens
                   if t["label"] == "ANSWER"
                   and abs(t["center_y"] - ty) < line_tol
                   and t["center_x"] > tx
                   and t["text"].upper() not in ("TOTAL", "TOTAL:")]
        if same_ln:
            best = min(same_ln, key=lambda t: t["center_x"])
            invoice["total_amount"] = parse_float(best["text"])
            logger.info(f"  TOTAL → {invoice['total_amount']}")

    # ── Pass 6: Fallback total from items sum ─────────────────────────────────
    if invoice["total_amount"] == 0.0 and invoice["items"]:
        invoice["total_amount"] = round(
            sum(item.get("amount", 0.0) for item in invoice["items"]), 2)

    # ── Pass 7: Date fallback ─────────────────────────────────────────────────
    if not invoice["invoice_date"]:
        for t in tokens:
            if re.match(r"\d{4}[/-]\d{2}[/-]\d{2}", t["text"]):
                invoice["invoice_date"] = t["text"]
                break

    return invoice


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="LiLT Invoice Inference r23")
    ap.add_argument("image")
    ap.add_argument("--model", "--model_dir", dest="model_dir", default="lilt_model_11")
    ap.add_argument("--output",        default="./results")
    ap.add_argument("--confidence",    type=float, default=0.15)
    ap.add_argument("--max-pair-dist", type=int,   default=50)
    ap.add_argument("--spatial",       type=int,   default=600)
    ap.add_argument("--no-relations",  action="store_true")
    ap.add_argument("--no-cuda",       action="store_true")
    ap.add_argument("--display",       action="store_true")
    ap.add_argument("--no-refine",     action="store_true")
    ap.add_argument("--conf-thresh",   type=float, default=0.0)
    args = ap.parse_args()

    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    config = InferenceConfig(
        model_dir=args.model_dir, device=device,
        refine_labels=not args.no_refine,
        confidence_threshold=args.conf_thresh,
        relation_confidence=args.confidence,
        max_pair_dist=args.max_pair_dist,
        spatial_dist=args.spatial,
    )

    print("=" * 70)
    print("🧾 LiLT Invoice Inference r23")
    print("=" * 70)
    print(f"📁 Image:      {args.image}")
    print(f"🤖 Model:      {args.model_dir}")
    print(f"⚙️  Device:     {device}")
    print(f"🔧 Refinement: {'ON' if config.refine_labels else 'OFF'}")
    print(f"🎚️  Conf thresh: {config.confidence_threshold}")
    print(f"🔗 Relations:  {'OFF' if args.no_relations else 'ON'}")
    if not args.no_relations:
        print(f"   Relation conf: {config.relation_confidence}")
        print(f"   Spatial dist:  {config.spatial_dist} px")
    print("=" * 70)

    if not Path(args.image).exists():
        logger.error(f"❌ Not found: {args.image}")
        sys.exit(1)

    extractor = LabelExtractor(config)
    result    = extractor.extract(args.image)

    if "error" in result:
        logger.error(f"Extraction failed: {result['error']}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    out_json = out_dir / f"{stem}_labels.json"
    with open(out_json, "w") as f:
        json.dump({
            "image":  result["image_path"],
            "size":   result["image_size"],
            "stats":  result["stats"],
            "tokens": [{"text": t["text"], "label": t["label"], "confidence": t["confidence"]}
                       for t in result["tokens"]],
        }, f, indent=2)
    logger.info(f"💾 Saved labels: {out_json}")

    invoice_data = build_invoice_json(result, args.image)
    invoice_path = out_dir / f"{stem}_invoice.json"
    with open(invoice_path, "w") as f:
        json.dump(invoice_data, f, indent=2)
    logger.info(f"💾 Saved invoice: {invoice_path}")

    if not args.no_relations:
        img_w, img_h = result["image_size"]["width"], result["image_size"]["height"]
        relations    = extract_relations(result["tokens"], config, img_w, img_h)
        rel_path     = out_dir / f"{stem}_relations.json"
        with open(rel_path, "w") as f:
            json.dump(relations, f, indent=2)
        logger.info(f"💾 Saved relations: {rel_path}")
        if relations:
            print("\n" + "=" * 70)
            print("🔗 Extracted Relations (QUESTION → ANSWER)")
            print("=" * 70)
            for r in relations:
                print(f"   '{r['question']}'  →  '{r['answer']}'  (conf={r['confidence']})")
            print("=" * 70)
        else:
            print("\n🔗 No relations found.\n")

    if result.get("visualization"):
        vis_path = out_dir / f"{stem}_visualization.png"
        result["visualization"].save(vis_path)
        logger.info(f"💾 Saved visualization: {vis_path}")
        if args.display:
            try:
                result["visualization"].show()
            except Exception as e:
                logger.warning(f"Display failed: {e}")

    s = result["stats"]
    print("\n" + "=" * 70)
    print("📊 Label Counts")
    print("=" * 70)
    print(f"   HEADER   : {s['HEADER']}")
    print(f"   QUESTION : {s['QUESTION']}")
    print(f"   ANSWER   : {s['ANSWER']}")
    print(f"   O        : {s['O']}")
    print("=" * 70)
    for label in ["HEADER", "QUESTION", "ANSWER"]:
        examples = [t for t in result["tokens"] if t["label"] == label][:5]
        if examples:
            print(f"\n{label} examples:")
            for ex in examples:
                print(f"   '{ex['text']}' (conf={ex['confidence']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
