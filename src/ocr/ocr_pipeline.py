"""
ContractOCR - EasyOCR-based OCR Pipeline for ContractIQ
=========================================================
Handles scanned PDFs and image files that PyMuPDF cannot extract text from.

Architecture:
    PDF Upload -> PyMuPDF text extraction -> if empty -> page images -> EasyOCR

Why EasyOCR over PaddleOCR:
    - Fits in 512MB RAM (free-tier deployment on Render/Koyeb)
    - Simple pip install (no PaddlePaddle framework)
    - 80+ language support

Usage:
    from src.ocr.ocr_pipeline import ContractOCR

    ocr = ContractOCR(languages=["en"])
    result = ocr.extract_text("scanned_contract.pdf")
    print(result.text, result.is_ocr, result.confidence)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import fitz  # PyMuPDF for PDF-to-image conversion

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from src.ocr.preprocessor import ImagePreprocessor

# Lazy-load EasyOCR to avoid import overhead on every request
_ocr_reader = None

# Supported image extensions for direct OCR
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# Min chars per page to consider extraction successful
MIN_TEXT_THRESHOLD = 50


@dataclass
class OCRPage:
    """OCR result for a single page."""
    page_number: int
    text: str
    confidence: float  # 0.0 - 1.0
    method: str  # "pymupdf" or "easyocr"


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    text: str
    pages: List[OCRPage] = field(default_factory=list)
    is_ocr: bool = False
    confidence: float = 1.0
    total_pages: int = 0
    ocr_pages: int = 0
    method: str = "pymupdf"  # "pymupdf", "easyocr", or "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "text": self.text,
            "is_ocr": self.is_ocr,
            "confidence": round(self.confidence, 3),
            "total_pages": self.total_pages,
            "ocr_pages": self.ocr_pages,
            "method": self.method,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text_length": len(p.text),
                    "confidence": round(p.confidence, 3),
                    "method": p.method,
                }
                for p in self.pages
            ],
        }


def _get_ocr_reader(languages: List[str]):
    """Lazy-load and cache the EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(
                languages, gpu=False, verbose=False,
            )
            log.info("EasyOCR reader initialized", languages=languages)
        except ImportError:
            log.error("EasyOCR not installed")
            raise DocumentPortalException(
                "EasyOCR required. Install: pip install easyocr", sys,
            )
    return _ocr_reader


class ContractOCR:
    """
    OCR pipeline for ContractIQ.

    Handles scanned PDFs, image files, and hybrid documents.
    Per-page assessment: PyMuPDF >= 50 chars -> use it; else -> EasyOCR.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        min_text_threshold: int = MIN_TEXT_THRESHOLD,
        dpi: int = 300,
    ):
        self.languages = languages or ["en"]
        self.min_text_threshold = min_text_threshold
        self.dpi = dpi
        self.preprocessor = ImagePreprocessor()
        log.info(
            "ContractOCR initialized",
            languages=self.languages,
            threshold=self.min_text_threshold,
            dpi=self.dpi,
        )

    def extract_text(self, file_path: str) -> OCRResult:
        """
        Extract text from a document file. Auto-detects PDF vs image.

        Args:
            file_path: Path to the document file

        Returns:
            OCRResult with text, per-page results, and confidence scores.
        """
        try:
            ext = Path(file_path).suffix.lower()
            if ext == ".pdf":
                return self._extract_from_pdf(str(file_path))
            elif ext in IMAGE_EXTENSIONS:
                return self._extract_from_image(str(file_path))
            else:
                raise ValueError(f"Unsupported OCR file type: {ext}")
        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("OCR extraction failed", file_path=file_path, error=str(e))
            raise DocumentPortalException(f"OCR failed: {file_path}", e) from e

    def _extract_from_pdf(self, pdf_path: str) -> OCRResult:
        """Extract text from PDF with automatic OCR fallback per page."""
        pages: List[OCRPage] = []
        ocr_page_count = 0

        with fitz.open(pdf_path) as doc:
            log.info("Processing PDF", path=pdf_path, pages=doc.page_count)

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pymupdf_text = page.get_text().strip()

                if len(pymupdf_text) >= self.min_text_threshold:
                    pages.append(OCRPage(
                        page_number=page_num + 1,
                        text=pymupdf_text,
                        confidence=1.0,
                        method="pymupdf",
                    ))
                else:
                    log.info(
                        "OCR fallback triggered",
                        page=page_num + 1,
                        chars=len(pymupdf_text),
                    )
                    pages.append(self._ocr_pdf_page(page, page_num + 1))
                    ocr_page_count += 1

        full_text = "\n\n".join(
            f"--- Page {p.page_number} ---\n{p.text}"
            for p in pages if p.text.strip()
        )
        avg_conf = sum(p.confidence for p in pages) / len(pages) if pages else 0.0

        if ocr_page_count == 0:
            method = "pymupdf"
        elif ocr_page_count == len(pages):
            method = "easyocr"
        else:
            method = "hybrid"

        result = OCRResult(
            text=full_text, pages=pages, is_ocr=ocr_page_count > 0,
            confidence=avg_conf, total_pages=len(pages),
            ocr_pages=ocr_page_count, method=method,
        )
        log.info(
            "PDF OCR complete", path=pdf_path,
            total=result.total_pages, ocr=result.ocr_pages,
            method=result.method, confidence=round(result.confidence, 3),
        )
        return result

    def _ocr_pdf_page(self, page, page_number: int) -> OCRPage:
        """Convert a single PDF page to image and run EasyOCR."""
        try:
            import numpy as np

            matrix = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pixmap = page.get_pixmap(matrix=matrix)
            img_array = np.frombuffer(
                pixmap.samples, dtype=np.uint8
            ).reshape(pixmap.height, pixmap.width, pixmap.n)

            if pixmap.n == 4:
                img_array = img_array[:, :, :3]

            processed = self.preprocessor.preprocess(img_array)
            reader = _get_ocr_reader(self.languages)
            results = reader.readtext(processed)

            text_parts = []
            confidences = []
            for _bbox, text, conf in results:
                if text.strip():
                    text_parts.append(text.strip())
                    confidences.append(conf)

            page_text = " ".join(text_parts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            log.info("OCR page done", page=page_number, chars=len(page_text),
                     confidence=round(avg_conf, 3))
            return OCRPage(page_number=page_number, text=page_text,
                          confidence=avg_conf, method="easyocr")
        except Exception as e:
            log.error("OCR page failed", page=page_number, error=str(e))
            return OCRPage(page_number=page_number, text="",
                          confidence=0.0, method="easyocr_failed")

    def _extract_from_image(self, image_path: str) -> OCRResult:
        """Extract text directly from an image file."""
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(image_path)
            img_array = np.array(img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]

            processed = self.preprocessor.preprocess(img_array)
            reader = _get_ocr_reader(self.languages)
            results = reader.readtext(processed)

            text_parts = []
            confidences = []
            for _bbox, text, conf in results:
                if text.strip():
                    text_parts.append(text.strip())
                    confidences.append(conf)

            page_text = " ".join(text_parts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            page = OCRPage(page_number=1, text=page_text,
                          confidence=avg_conf, method="easyocr")

            log.info("Image OCR complete", path=image_path,
                     chars=len(page_text), confidence=round(avg_conf, 3))
            return OCRResult(
                text=page_text, pages=[page], is_ocr=True,
                confidence=avg_conf, total_pages=1,
                ocr_pages=1, method="easyocr",
            )
        except Exception as e:
            log.error("Image OCR failed", path=image_path, error=str(e))
            raise DocumentPortalException(f"Image OCR failed: {image_path}", e) from e

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """Quick check if a PDF is scanned (first 3 pages)."""
        try:
            with fitz.open(pdf_path) as doc:
                n = min(3, doc.page_count)
                chars = sum(len(doc.load_page(i).get_text().strip()) for i in range(n))
                return (chars / n if n else 0) < self.min_text_threshold
        except Exception:
            return False
