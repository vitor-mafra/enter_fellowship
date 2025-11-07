"""Text extraction from PDF documents."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import fitz

logger = logging.getLogger(__name__)


def extract_text_simple(pdf_path: str) -> str:
    """Extract plain text from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Concatenated plain text from all pages.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF file is invalid or unreadable.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        all_text = []
        
        for page_num in range(page_count):
            page = doc[page_num]
            all_text.append(page.get_text("text"))
        
        doc.close()
        full_text = "\n".join(all_text)
        
        logger.info(
            f"Extracted text from {pdf_path}: {page_count} pages, {len(full_text)} characters"
        )
        
        return full_text
        
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF {pdf_path}: {str(e)}") from e


def extract_text_with_coords(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text blocks with bounding boxes from PDF.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of dictionaries, each containing:
        - text: Extracted text content
        - x0, y0, x1, y1: Bounding box coordinates
        - page: Page number (0-indexed)
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF file is invalid or unreadable.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        all_blocks = []
        
        for page_num in range(page_count):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    block_text_parts = []
                    min_x = float("inf")
                    min_y = float("inf")
                    max_x = float("-inf")
                    max_y = float("-inf")
                    
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_content = span.get("text", "").strip()
                            if text_content:
                                block_text_parts.append(text_content)
                                bbox = span.get("bbox", [])
                                if len(bbox) >= 4:
                                    min_x = min(min_x, bbox[0])
                                    min_y = min(min_y, bbox[1])
                                    max_x = max(max_x, bbox[2])
                                    max_y = max(max_y, bbox[3])
                    
                    if block_text_parts and min_x != float("inf"):
                        all_blocks.append({
                            "text": " ".join(block_text_parts),
                            "x0": float(min_x),
                            "y0": float(min_y),
                            "x1": float(max_x),
                            "y1": float(max_y),
                            "page": page_num,
                        })
        
        doc.close()
        
        logger.info(
            f"Extracted text blocks from {pdf_path}: {page_count} pages, {len(all_blocks)} blocks"
        )
        
        return all_blocks
        
    except Exception as e:
        raise ValueError(
            f"Failed to extract text blocks from PDF {pdf_path}: {str(e)}"
        ) from e

