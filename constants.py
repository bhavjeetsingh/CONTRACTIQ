"""
Global constants for ContractIQ application.
"""

# Supported file extensions for document processing
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".ppt", ".pptx",
    ".xlsx", ".xls", ".csv", ".json", ".rtf",
    # Image formats — processed via OCR pipeline
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
}

# Image-only extensions (routed directly to OCR, not text extraction)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# Maximum upload file size in bytes (50MB)
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

# OCR configuration
OCR_MIN_TEXT_THRESHOLD = 50  # Min chars per page before OCR fallback triggers
OCR_DEFAULT_DPI = 300  # Resolution for PDF-to-image conversion
OCR_DEFAULT_LANGUAGES = ["en"]  # Default OCR language(s)
