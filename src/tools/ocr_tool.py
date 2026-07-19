"""
OCR Tool for Medical Documents
Extracts text from PDFs, images, and scanned lab reports
Uses pytesseract + PIL for OCR, pypdf for PDFs
"""

import os
import re
import base64
import tempfile
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_file(file_path: str) -> dict:
    """
    Extract text from medical documents.
    Supports: PDF, PNG, JPG, JPEG, TIFF
    Returns: {text, method, page_count, confidence}
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
        return _extract_image_ocr(file_path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {"text": text, "method": "text", "page_count": 1, "confidence": 1.0}
    else:
        raise ValueError(f"Unsupported format: {ext}")


def _extract_pdf(path: str) -> dict:
    """Extract text from PDF — tries text extraction first, OCR fallback"""
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        pages = []
        for page in reader.pages[:50]:  # max 50 pages
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)

        if pages:
            return {
                "text": "\n\n".join(pages),
                "method": "pdf_text",
                "page_count": len(reader.pages),
                "confidence": 0.95,
            }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PDF text extraction failed: {e}")

    # Fallback: OCR each page
    return _ocr_pdf_pages(path)


def _ocr_pdf_pages(path: str) -> dict:
    """Convert PDF pages to images and OCR them"""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(path, dpi=200, first_page=1, last_page=10)
        texts = []
        for img in images:
            result = _ocr_image(img)
            if result:
                texts.append(result)
        return {
            "text": "\n\n".join(texts),
            "method": "pdf_ocr",
            "page_count": len(images),
            "confidence": 0.85,
        }
    except ImportError:
        return {"text": "PDF OCR requires pdf2image and poppler. Text extraction only.", "method": "fallback", "page_count": 1, "confidence": 0.5}
    except Exception as e:
        return {"text": f"OCR error: {str(e)}", "method": "error", "page_count": 0, "confidence": 0.0}


def _extract_image_ocr(path: str) -> dict:
    """OCR a single image file"""
    try:
        from PIL import Image
        img = Image.open(path)
        text = _ocr_image(img)
        return {
            "text": text or "",
            "method": "image_ocr",
            "page_count": 1,
            "confidence": 0.85,
        }
    except ImportError:
        return {"text": "PIL not installed", "method": "error", "page_count": 0, "confidence": 0.0}


def _ocr_image(img) -> str:
    """Run Tesseract OCR on a PIL image"""
    try:
        import pytesseract
        from PIL import ImageFilter, ImageEnhance

        # Preprocess for better OCR accuracy
        img = img.convert("L")           # Grayscale
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(img, config="--psm 6")
        return _clean_ocr_text(text)
    except ImportError:
        logger.warning("pytesseract not installed — OCR unavailable")
        return ""
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return ""


def _extract_docx(path: str) -> dict:
    try:
        from docx import Document
        doc = Document(path)
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return {"text": text, "method": "docx", "page_count": 1, "confidence": 0.95}
    except ImportError:
        return {"text": "python-docx not installed", "method": "error", "page_count": 0, "confidence": 0.0}


def _clean_ocr_text(text: str) -> str:
    """Clean common OCR artifacts"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u0900-\u097F]', '', text)
    return text.strip()


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for vision API"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_document_type(text: str) -> str:
    """Detect what kind of medical document this is"""
    text_lower = text.lower()

    patterns = {
        "lab_report": ["hemoglobin", "platelet", "wbc", "rbc", "glucose", "creatinine", "cholesterol", "reference range", "normal range"],
        "prescription": ["rx", "refill", "sig:", "dispense", "tablet", "capsule", "mg", "dose", "twice daily", "once daily", "prescription"],
        "radiology_report": ["x-ray", "mri", "ct scan", "ultrasound", "impression:", "findings:", "radiologist"],
        "discharge_summary": ["discharge", "admitted", "diagnosis", "hospital", "inpatient", "follow up"],
        "pathology_report": ["biopsy", "specimen", "pathology", "malignant", "benign", "histology"],
    }

    scores = {}
    for doc_type, keywords in patterns.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = score

    if max(scores.values(), default=0) == 0:
        return "general_medical"

    return max(scores, key=scores.get)
