"""
loader.py — Stage 1: Document Loading

Responsibility: Read a plain text file from disk and return its contents as a string.

Why a separate module?
  In later stages, this will grow to support PDFs, URLs, etc.
  Keeping it isolated means we swap the loader without touching anything else.
"""


def load_text_file(filepath: str) -> str:
    """
    Read a plain .txt file and return its contents as a string.

    Args:
        filepath: Absolute or relative path to the .txt file.

    Returns:
        The full text content of the file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"[Loader] Loaded '{filepath}' — {len(content)} characters")
    return content
