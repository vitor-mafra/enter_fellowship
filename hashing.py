"""Perceptual hashing utilities for template recognition."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import imagehash
from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger(__name__)

FULL_THRESHOLD = 20
HALF_THRESHOLD = 15
TARGET_WIDTH = 800
TARGET_HEIGHT = 1131
DPI = 100
HASH_SIZE = 8

_template_cache: Dict[str, Dict[str, int]] = {}


def _render_pdf_deterministic(pdf_path: str, page_num: int = 0) -> Image.Image:
    """Render PDF page with fixed parameters for deterministic output.
    
    Args:
        pdf_path: Path to the PDF file.
        page_num: Page number to render (0-indexed).
        
    Returns:
        PIL Image in grayscale, normalized size.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        pix = page.get_pixmap(
            dpi=DPI,
            colorspace=fitz.csGRAY,
            alpha=False,
            annots=False,
            clip=None
        )
        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BILINEAR)
        img = ImageOps.expand(img, border=2, fill=255)
        return img
    finally:
        doc.close()


def _mask_variable_regions(img: Image.Image, mask_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> Image.Image:
    """Mask out variable regions (photos, stamps, etc.) before hashing.
    
    Args:
        img: PIL Image to mask.
        mask_regions: List of (x0, y0, x1, y1) rectangles to mask. Defaults to typical photo zones.
        
    Returns:
        Image with masked regions filled with white.
    """
    if mask_regions is None:
        mask_regions = [
            (80, 120, 240, 360),
            (560, 120, 720, 360),
        ]
    
    img_masked = img.copy()
    draw = ImageDraw.Draw(img_masked)
    
    for x0, y0, x1, y1 in mask_regions:
        draw.rectangle((x0, y0, x1, y1), fill=255)
    
    return img_masked


def compute_diff_hashes_stable(pdf_path: str, page: int = 0) -> Dict[str, int]:
    """Compute full/left/right dhashes from a masked, normalized grayscale render.
    
    Args:
        pdf_path: Path to the PDF file.
        page: Page number to hash (0-indexed, defaults to 0).
        
    Returns:
        Dictionary with keys "full", "left", "right" containing integer hashes.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF is invalid or page number is out of range.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    cache_key = f"{pdf_path}:{page}"
    if cache_key in _template_cache:
        return _template_cache[cache_key]
    
    try:
        img = _render_pdf_deterministic(pdf_path, page_num=page)
        img_masked = _mask_variable_regions(img)
        
        width, height = img_masked.size
        
        hash_full = imagehash.dhash(img_masked, hash_size=HASH_SIZE)
        hash_full_int = int(str(hash_full), 16)
        
        left_half = img_masked.crop((0, 0, width // 2, height))
        hash_left = imagehash.dhash(left_half, hash_size=HASH_SIZE)
        hash_left_int = int(str(hash_left), 16)
        
        right_half = img_masked.crop((width // 2, 0, width, height))
        hash_right = imagehash.dhash(right_half, hash_size=HASH_SIZE)
        hash_right_int = int(str(hash_right), 16)
        
        result = {
            "full": hash_full_int,
            "left": hash_left_int,
            "right": hash_right_int
        }
        
        _template_cache[cache_key] = result
        
        logger.info(
            f"Computed stable hashes for {pdf_path} (page {page}): "
            f"full={hash_full_int}, left={hash_left_int}, right={hash_right_int}"
        )
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to compute stable hashes for {pdf_path}: {str(e)}") from e


def compare_diff_hashes_stable(
    hash_a: Dict[str, int],
    hash_b: Dict[str, int],
    full_threshold: int = FULL_THRESHOLD,
    half_threshold: int = HALF_THRESHOLD,
) -> Tuple[bool, Dict[str, int]]:
    """Compare two hash sets using the heuristic and return (same_template, distances).
    
    Heuristic rule:
    - diff_full < full_threshold AND (diff_left < half_threshold OR diff_right < half_threshold)
    → same template
    
    Args:
        hash_a: First document's hash set (dict with "full", "left", "right" as ints).
        hash_b: Second document's hash set (dict with "full", "left", "right" as ints).
        full_threshold: Maximum Hamming distance for full page (default: 80).
        half_threshold: Maximum Hamming distance for half pages (default: 40).
        
    Returns:
        Tuple of (is_match, distances_dict) where distances_dict contains "full", "left", "right".
    """
    diff_full = bin(hash_a["full"] ^ hash_b["full"]).count("1")
    diff_left = bin(hash_a["left"] ^ hash_b["left"]).count("1")
    diff_right = bin(hash_a["right"] ^ hash_b["right"]).count("1")
    
    distances = {
        "full": diff_full,
        "left": diff_left,
        "right": diff_right
    }
    
    condition_full = diff_full < full_threshold
    condition_half = (diff_left < half_threshold) or (diff_right < half_threshold)
    is_match = condition_full and condition_half
    
    logger.info(
        f"Hash comparison: full={diff_full}, left={diff_left}, right={diff_right}, "
        f"match={is_match}"
    )
    
    return (is_match, distances)


def compute_diff_hashes(pdf_path: str, page: int = 0, hash_size: int = 16) -> Dict[str, str]:
    """Compute multi-region diff hashes for a PDF page.
    
    Legacy function for backward compatibility. Converts integer hashes to hex strings.
    
    Args:
        pdf_path: Path to the PDF file.
        page: Page number to hash (0-indexed, defaults to 0).
        hash_size: Legacy parameter (ignored, uses HASH_SIZE=8).
        
    Returns:
        Dictionary with keys "full", "left", "right" containing hex string hashes.
    """
    int_hashes = compute_diff_hashes_stable(pdf_path, page)
    return {
        "full": format(int_hashes["full"], "x"),
        "left": format(int_hashes["left"], "x"),
        "right": format(int_hashes["right"], "x")
    }


def compare_diff_hashes(
    hash_a: Dict[str, str],
    hash_b: Dict[str, str],
    full_threshold: int = FULL_THRESHOLD,
    half_threshold: int = HALF_THRESHOLD,
) -> bool:
    """Compare two multi-region hash sets using heuristic rule.
    
    Legacy function for backward compatibility. Converts hex strings to ints.
    
    Args:
        hash_a: First document's hash set (dict with "full", "left", "right" as hex strings).
        hash_b: Second document's hash set (dict with "full", "left", "right" as hex strings).
        full_threshold: Maximum Hamming distance for full page (default: 80).
        half_threshold: Maximum Hamming distance for half pages (default: 40).
        
    Returns:
        True if documents belong to the same template, False otherwise.
    """
    int_hash_a = {
        "full": int(hash_a["full"], 16) if isinstance(hash_a["full"], str) else hash_a["full"],
        "left": int(hash_a["left"], 16) if isinstance(hash_a["left"], str) else hash_a["left"],
        "right": int(hash_a["right"], 16) if isinstance(hash_a["right"], str) else hash_a["right"],
    }
    int_hash_b = {
        "full": int(hash_b["full"], 16) if isinstance(hash_b["full"], str) else hash_b["full"],
        "left": int(hash_b["left"], 16) if isinstance(hash_b["left"], str) else hash_b["left"],
        "right": int(hash_b["right"], 16) if isinstance(hash_b["right"], str) else hash_b["right"],
    }
    
    is_match, _ = compare_diff_hashes_stable(int_hash_a, int_hash_b, full_threshold, half_threshold)
    return is_match


def find_similar_template(
    pdf_path: str,
    known_templates: Dict[str, Dict[str, str]],
    label: Optional[str] = None,
    full_threshold: int = FULL_THRESHOLD,
    half_threshold: int = HALF_THRESHOLD,
) -> Optional[str]:
    """Find if a PDF matches any known template using multi-region hashing.
    
    Only considers templates with the same label. Uses heuristic comparison
    rule: diff_full < 80 AND (diff_left < 40 OR diff_right < 40).
    
    Args:
        pdf_path: Path to the PDF file to match.
        known_templates: Dictionary mapping template IDs to hash sets
                        containing "full", "left", "right" hashes and "label".
        label: Optional document label to filter templates (only compare with same label).
        full_threshold: Maximum Hamming distance for full page (default: 80).
        half_threshold: Maximum Hamming distance for half pages (default: 40).
        
    Returns:
        Template ID of the most similar template if found, None otherwise.
    """
    try:
        current_hashes_int = compute_diff_hashes_stable(pdf_path)
        current_hashes = {
            "full": format(current_hashes_int["full"], "x"),
            "left": format(current_hashes_int["left"], "x"),
            "right": format(current_hashes_int["right"], "x")
        }
    except Exception as e:
        logger.error(f"Failed to compute hashes for {pdf_path}: {str(e)}")
        return None
    
    best_match = None
    best_score = float('inf')
    
    for template_id, template_data in known_templates.items():
        template_label = template_data.get("label")
        template_hashes = {
            "full": template_data.get("hash_full", ""),
            "left": template_data.get("hash_left", ""),
            "right": template_data.get("hash_right", ""),
        }
        
        if not template_hashes["full"]:
            continue
        
        if label and template_label != label:
            continue
        
        if not template_hashes["left"] or not template_hashes["right"]:
            try:
                hash_full_a = int(current_hashes["full"], 16)
                hash_full_b = int(template_hashes["full"], 16)
                diff_full = bin(hash_full_a ^ hash_full_b).count("1")
                is_match = diff_full < full_threshold
            except (ValueError, TypeError):
                continue
        else:
            is_match = compare_diff_hashes(
                current_hashes,
                template_hashes,
                full_threshold=full_threshold,
                half_threshold=half_threshold,
            )
        
        if is_match:
            try:
                hash_full_a = int(current_hashes["full"], 16)
                hash_full_b = int(template_hashes["full"], 16)
                diff_full = bin(hash_full_a ^ hash_full_b).count("1")
                
                if template_hashes["left"] and template_hashes["right"]:
                    hash_left_a = int(current_hashes["left"], 16)
                    hash_left_b = int(template_hashes["left"], 16)
                    hash_right_a = int(current_hashes["right"], 16)
                    hash_right_b = int(template_hashes["right"], 16)
                    diff_left = bin(hash_left_a ^ hash_left_b).count("1")
                    diff_right = bin(hash_right_a ^ hash_right_b).count("1")
                    score = diff_full + min(diff_left, diff_right)
                else:
                    score = diff_full
                
                if score < best_score:
                    best_score = score
                    best_match = template_id
            except (ValueError, TypeError):
                continue
    
    if best_match:
        logger.info(
            f"Found similar template using stable multi-region hashing: {best_match} "
            f"(score: {best_score})"
        )
        return best_match
    
    return None


def compute_phash(pdf_path: str, page: int = 0, hash_size: int = 16) -> str:
    """Compute a difference hash (dHash) for a PDF page.
    
    Legacy function for backward compatibility. Returns only the full hash.
    Use compute_diff_hashes_stable for new code.
    
    Args:
        pdf_path: Path to the PDF file.
        page: Page number to hash (0-indexed, defaults to 0).
        hash_size: Legacy parameter (ignored).
        
    Returns:
        Hexadecimal string representing the perceptual hash (full page only).
    """
    hashes = compute_diff_hashes_stable(pdf_path, page)
    return format(hashes["full"], "x")


def compare_phashes(hash1: str, hash2: str, verbose: bool = True) -> int:
    """Compare two perceptual hashes and return their Hamming distance.
    
    Legacy function for backward compatibility.
    
    Args:
        hash1: First perceptual hash (hexadecimal string or int).
        hash2: Second perceptual hash (hexadecimal string or int).
        verbose: Whether to print comparison details.
        
    Returns:
        Hamming distance between the two hashes (0 = identical).
    """
    try:
        int1 = int(hash1, 16) if isinstance(hash1, str) else hash1
        int2 = int(hash2, 16) if isinstance(hash2, str) else hash2
        distance = bin(int1 ^ int2).count("1")
        return distance
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to compare hashes: {str(e)}")
        return 999


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Calculate Hamming distance between two hexadecimal hash strings.
    
    Args:
        hex_hash1: First hash as hexadecimal string or int.
        hex_hash2: Second hash as hexadecimal string or int.
        
    Returns:
        Hamming distance (number of differing bits).
    """
    try:
        int1 = int(hex_hash1, 16) if isinstance(hex_hash1, str) else hex_hash1
        int2 = int(hex_hash2, 16) if isinstance(hex_hash2, str) else hex_hash2
        return bin(int1 ^ int2).count("1")
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to calculate Hamming distance: {str(e)}")
        return 999


def store_template_hash(pdf_path: str, hash_value: Optional[str] = None) -> str:
    """Store a template hash for future matching.
    
    Legacy function for backward compatibility.
    
    Args:
        pdf_path: Path to the PDF file.
        hash_value: Optional perceptual hash value. If None, computes it.
        
    Returns:
        The stored/computed hash value (full page only).
    """
    if hash_value is None:
        hash_value = compute_phash(pdf_path)
    
    cache_key = f"{pdf_path}:0"
    hashes = compute_diff_hashes_stable(pdf_path)
    _template_cache[cache_key] = hashes
    logger.debug(f"Stored template hash for {pdf_path}: {hash_value}")
    return hash_value


def average_hashes(hashes: List[str]) -> str:
    """Calculate the average of multiple perceptual hashes.
    
    Args:
        hashes: List of perceptual hashes (hexadecimal strings or ints).
        
    Returns:
        Hexadecimal string representing the averaged hash.
    """
    if not hashes:
        raise ValueError("Cannot average empty list of hashes")
    
    if len(hashes) == 1:
        return hashes[0] if isinstance(hashes[0], str) else format(hashes[0], "x")
    
    try:
        import numpy as np
        int_hashes = [int(h, 16) if isinstance(h, str) else h for h in hashes]
        hash_objects = [imagehash.hex_to_hash(format(h, "x")) for h in int_hashes]
        hash_arrays = [np.array(h.hash.flatten(), dtype=np.float64) for h in hash_objects]
        avg_arr = np.mean(hash_arrays, axis=0)
        avg_arr = np.round(avg_arr).astype(np.uint8)
        avg_hash_array = avg_arr.reshape(hash_objects[0].hash.shape)
        avg_hash = imagehash.ImageHash(avg_hash_array)
        return str(avg_hash)
    except Exception as e:
        logger.error(f"Failed to average hashes: {str(e)}")
        return hashes[0] if isinstance(hashes[0], str) else format(hashes[0], "x")
