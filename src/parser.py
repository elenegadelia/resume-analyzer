"""
parser.py — Extract raw text from resume files.

Supports: PDF (.pdf), Word (.docx), and plain text (.txt)

Why keep this separate? Because parsing (reading a file) is completely different
from analyzing text. Clean separation makes code easier to test and maintain.
"""

import fitz  # PyMuPDF — for reading PDF files
from docx import Document  # python-docx — for reading Word files


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    PyMuPDF reads each page of the PDF and pulls out the raw text.
    Some PDFs (scanned images) won't have selectable text — we handle that gracefully.
    """
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        raise ValueError(f"Could not read PDF file: {e}")
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract all text from a .docx Word document.

    Word documents are structured in paragraphs. We join them with newlines.
    """
    try:
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Could not read DOCX file: {e}")


def extract_text_from_txt(file_path: str) -> str:
    """Read a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Could not read text file: {e}")


def extract_text(file_path: str) -> str:
    """
    Main entry point. Detects the file type by its extension and calls
    the correct extraction function.

    Args:
        file_path: path to the resume file

    Returns:
        raw text extracted from the file
    """
    lower = file_path.lower()

    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif lower.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from in-memory file bytes (used by Streamlit file uploader).

    Streamlit gives us the file as bytes, not a path on disk.
    We write it to a temp file, extract, then clean up.
    """
    import tempfile
    import os

    suffix = "." + filename.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path)
    finally:
        os.unlink(tmp_path)  # always delete the temp file

    return text
