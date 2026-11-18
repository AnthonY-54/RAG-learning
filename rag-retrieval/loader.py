"""
loader.py — Stage 2: Multi-Format Document Loading

Supports:
  1. Plain text (.txt)
  2. PDF documents (.pdf) via pypdf

Returns:
  Clean extracted text and metadata (file type, character count, page count if PDF).
"""

import os
from pypdf import PdfReader


def load_document(filepath: str) -> tuple[str, dict]:
    """
    Load a document (.txt or .pdf) and return extracted text and metadata.

    Args:
        filepath: Path to the document.

    Returns:
        tuple (text_content, metadata_dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        metadata = {
            "source": os.path.basename(filepath),
            "file_type": "txt",
            "pages": 1,
            "char_count": len(text),
        }
        print(f"[Loader] Loaded TXT '{metadata['source']}' — {len(text)} chars")
        return text, metadata

    elif ext == ".pdf":
        reader = PdfReader(filepath)
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_content = page.extract_text() or ""
            pages_text.append(page_content)

        full_text = "\n\n".join(pages_text)
        metadata = {
            "source": os.path.basename(filepath),
            "file_type": "pdf",
            "pages": len(reader.pages),
            "char_count": len(full_text),
        }
        print(f"[Loader] Loaded PDF '{metadata['source']}' — {len(reader.pages)} page(s), {len(full_text)} chars")
        return full_text, metadata

    else:
        raise ValueError(f"Unsupported file format '{ext}'. Only .txt and .pdf are supported.")
