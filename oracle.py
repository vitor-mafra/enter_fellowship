"""Oracle module for LLM-based extraction and template creation.

This module uses the LLM to extract data from documents and creates
optimized templates that store both extracted values and positional
information (gabarito) for fast future extraction of similar documents.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_utils import extract_data_with_llm
from template_manager import (
    find_template_by_label_and_fields,
    update_template,
    generate_template_id,
    learn_box_pattern as learn_template_pattern,
    learn_split_rules,
)
from box_parser import segment_box_text, learn_box_pattern, infer_segment_type
from hashing import compute_diff_hashes

logger = logging.getLogger(__name__)

# Template storage: hash -> template data
# Structure: {
#   "hash": {
#     "hash": str,
#     "extracted_data": Dict[str, Any],  # Ground truth values
#     "field_positions": Dict[str, Dict[str, Any]],  # Gabarito for positional extraction
#     "extraction_schema": Dict[str, str],  # Schema used
#     "pdf_path": str,  # Reference PDF
#   }
# }
_templates: Dict[str, Dict[str, Any]] = {}


def _text_blocks_to_simple_text(text_blocks: List[Dict[str, Any]]) -> str:
    """Convert text blocks with coordinates to simple text string.
    
    Args:
        text_blocks: List of text blocks with coordinates.
        
    Returns:
        Simple text string with all text content.
    """
    return "\n".join(
        [block.get("text", "").strip() for block in text_blocks if block.get("text", "").strip()]
    )


def _extract_field_positions(
    text_blocks: List[Dict[str, Any]], extracted_data: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Extract bounding box positions for each field based on extracted text.
    
    This creates the "gabarito" (ground truth positions) for positional extraction.
    Also detects if the extracted value is EXACTLY the box content or just PART of it.
    When multiple fields map to the same box, learns extraction patterns.
    
    Args:
        text_blocks: List of text blocks with coordinates from PDF.
        extracted_data: Dictionary of extracted field values (ground truth).
        
    Returns:
        Tuple of:
        - Dictionary mapping field names to their bounding box coordinates (gabarito).
        - Dictionary mapping field names to extraction mode ("exact" or "partial").
        - Dictionary mapping box text to list of fields that map to it (for pattern learning).
    """
    field_positions: Dict[str, Dict[str, Any]] = {}
    extraction_modes: Dict[str, str] = {}
    box_to_fields: Dict[str, List[Tuple[str, str]]] = {}  # box_text -> [(field_name, field_value), ...]
    
    for field_name, field_value in extracted_data.items():
        if field_value is None:
            continue
        
        field_value_str = str(field_value).strip()
        field_value_lower = field_value_str.lower()
        
        for block in text_blocks:
            block_text = block.get("text", "").strip()
            block_text_lower = block_text.lower()
            
            # Check for exact match (case-insensitive, ignoring extra whitespace)
            if field_value_lower == block_text_lower:
                field_positions[field_name] = {
                    "x0": block.get("x0"),
                    "y0": block.get("y0"),
                    "x1": block.get("x1"),
                    "y1": block.get("y1"),
                    "page": block.get("page", 0),
                }
                extraction_modes[field_name] = "exact"
                logger.debug(
                    f"Field '{field_name}': EXACT match with box '{block_text[:50]}...'"
                )
                # Track which fields map to this box
                if block_text not in box_to_fields:
                    box_to_fields[block_text] = []
                box_to_fields[block_text].append((field_name, field_value_str))
                break
            # Check for partial match (value is substring of box, or box is substring of value)
            elif field_value_lower in block_text_lower or block_text_lower in field_value_lower:
                # Determine if it's truly partial (not just whitespace differences)
                if len(field_value_lower) < len(block_text_lower) * 0.9:
                    field_positions[field_name] = {
                        "x0": block.get("x0"),
                        "y0": block.get("y0"),
                        "x1": block.get("x1"),
                        "y1": block.get("y1"),
                        "page": block.get("page", 0),
                    }
                    extraction_modes[field_name] = "partial"
                    logger.debug(
                        f"Field '{field_name}': PARTIAL match - value '{field_value_str}' "
                        f"is part of box '{block_text[:50]}...'"
                    )
                    # Track which fields map to this box
                    if block_text not in box_to_fields:
                        box_to_fields[block_text] = []
                    box_to_fields[block_text].append((field_name, field_value_str))
                    break
    
    return field_positions, extraction_modes, box_to_fields


def extract_with_llm(
    text_blocks: List[Dict[str, Any]],
    extraction_schema: Dict[str, str],
    fields_to_extract: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract data from document using LLM (Oracle).
    
    This is the main function that uses the LLM to create ground truth extraction.
    It converts text blocks to simple text and sends to LLM for extraction.
    
    Args:
        text_blocks: List of text blocks with coordinates from PDF.
        extraction_schema: Dictionary mapping field names to extraction descriptions.
        fields_to_extract: Optional list of field names to extract. If None, extracts all fields.
                          This allows incremental extraction (only missing fields).
        
    Returns:
        Dictionary with extracted field values (ground truth).
    """
    text_content = _text_blocks_to_simple_text(text_blocks)
    
    # If fields_to_extract is specified, create a subset schema
    if fields_to_extract:
        subset_schema = {
            field: extraction_schema[field]
            for field in fields_to_extract
            if field in extraction_schema
        }
        if not subset_schema:
            return {}
        extracted_data = extract_data_with_llm(text_content, subset_schema)
    else:
        extracted_data = extract_data_with_llm(text_content, extraction_schema)
    
    return extracted_data


def create_template(
    template_hash: str,
    text_blocks: List[Dict[str, Any]],
    extracted_data: Dict[str, Any],
    extraction_schema: Dict[str, str],
    pdf_path: str,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and store an extraction template.
    
    A template contains:
    - Hash identifier
    - Extracted data (ground truth values)
    - Field positions (gabarito for positional extraction)
    - Extraction schema
    - Reference PDF path
    
    Args:
        template_hash: Perceptual hash of the template.
        text_blocks: Text blocks with coordinates from PDF.
        extracted_data: Extracted field values (ground truth from LLM).
        extraction_schema: Schema used for extraction.
        pdf_path: Path to the reference PDF.
        label: Document label/type (optional, for template_manager integration).
        
    Returns:
        The created template dictionary.
    """
    # Extract field positions (gabarito) from text blocks
    # Also detect if values are EXACT or PARTIAL matches with boxes
    field_positions, extraction_modes, box_to_fields = _extract_field_positions(text_blocks, extracted_data)
    
    # Learn extraction patterns for boxes that contain multiple fields
    learned_patterns = {}
    learned_delimiters = {}
    
    for box_text, field_list in box_to_fields.items():
        if len(field_list) > 1:
            # Multiple fields map to the same box - need to learn pattern
            logger.info(
                f"Learning extraction pattern for box with {len(field_list)} fields: "
                f"'{box_text[:50]}...'"
            )
            
            # Try to segment the box text
            segments = segment_box_text(box_text)
            
            if len(segments) >= len(field_list):
                # Map fields to segments by finding best matches
                # Create a mapping: field_name -> segment_index
                # Use greedy assignment to avoid conflicts
                field_to_segment = {}
                used_segments = set()
                field_values = {field_name: field_value for field_name, field_value in field_list}
                
                # Score all field-segment pairs
                field_segment_scores = []
                for field_name, field_value in field_list:
                    field_value_lower = str(field_value).strip().lower()
                    
                    for seg_idx, segment in enumerate(segments):
                        segment_lower = segment.lower()
                        # Check if field value is contained in segment or vice versa
                        if field_value_lower in segment_lower or segment_lower in field_value_lower:
                            # Score based on length similarity and exact match
                            length_ratio = min(len(field_value_lower), len(segment_lower)) / max(len(field_value_lower), len(segment_lower), 1)
                            # Bonus for exact match
                            exact_match_bonus = 0.5 if field_value_lower == segment_lower else 0.0
                            score = length_ratio + exact_match_bonus
                            field_segment_scores.append((field_name, seg_idx, score))
                
                # Sort by score (descending) and assign greedily
                field_segment_scores.sort(key=lambda x: x[2], reverse=True)
                
                for field_name, seg_idx, score in field_segment_scores:
                    if field_name not in field_to_segment and seg_idx not in used_segments:
                        field_to_segment[field_name] = seg_idx
                        used_segments.add(seg_idx)
                        logger.debug(
                            f"Mapped field '{field_name}' to segment {seg_idx} "
                            f"('{segments[seg_idx][:30]}...') with score {score:.3f}"
                        )
                
                # Learn the type pattern
                pattern = learn_box_pattern(segments)
                
                # Create field-to-segment mapping for extraction
                field_segment_map = {}
                for field_name, seg_idx in field_to_segment.items():
                    field_segment_map[field_name] = {
                        "segment_index": seg_idx,
                        "segment_text": segments[seg_idx],
                        "segment_type": pattern[seg_idx] if seg_idx < len(pattern) else "text",
                    }
                
                learned_patterns[box_text] = {
                    "pattern": pattern,
                    "segments": segments,
                    "fields": [field_name for field_name, _ in field_list],
                    "field_to_segment": field_to_segment,  # field_name -> segment_index
                    "field_segment_map": field_segment_map,  # Detailed mapping
                }
                logger.info(
                    f"Learned pattern for box: {pattern} "
                    f"(fields: {[f[0] for f in field_list]}, "
                    f"mapping: {field_to_segment})"
                )
                
                # Try to identify effective delimiters
                # Check which delimiters successfully split the box
                from box_parser import DEFAULT_DELIMITERS
                import re
                for delimiter in DEFAULT_DELIMITERS:
                    if re.search(delimiter, box_text):
                        if box_text not in learned_delimiters:
                            learned_delimiters[box_text] = []
                        learned_delimiters[box_text].append(delimiter)
    
    # Compute multi-region hashes for template matching
    try:
        multi_region_hashes = compute_diff_hashes(pdf_path)
        hash_full = multi_region_hashes["full"]
        hash_left = multi_region_hashes["left"]
        hash_right = multi_region_hashes["right"]
    except Exception as e:
        logger.warning(f"Failed to compute multi-region hashes: {str(e)}, using legacy hash")
        hash_full = template_hash
        hash_left = ""
        hash_right = ""
    
    template = {
        "hash": template_hash,
        "hash_full": hash_full,
        "hash_left": hash_left,
        "hash_right": hash_right,
        "extracted_data": extracted_data,
        "field_positions": field_positions,
        "extraction_modes": extraction_modes,  # "exact" or "partial" for each field
        "extraction_schema": extraction_schema,
        "pdf_path": pdf_path,
        "box_patterns": learned_patterns,  # Patterns learned for multi-field boxes
        "box_delimiters": learned_delimiters,  # Effective delimiters for each box
        "label": label,  # Store label for filtering
    }
    
    _templates[template_hash] = template
    
    logger.info(
        f"Created template {template_hash} with {len(field_positions)} field positions "
        f"and {len(learned_patterns)} learned patterns"
    )
    
    # Update persistent template knowledge (only for stricted documents with label)
    if label:
        try:
            field_names = list(extraction_schema.keys())
            template_id = generate_template_id(label, field_names)
            update_template(template_id, label, extracted_data, extraction_modes=extraction_modes)
            
            # Learn patterns in template_manager for persistent storage
            for box_text, pattern_data in learned_patterns.items():
                pattern_sequence = pattern_data["pattern"]
                learn_template_pattern(template_id, label, pattern_sequence)
            
            # Learn split rules (delimiters)
            for box_text, delimiters in learned_delimiters.items():
                if delimiters:
                    learn_split_rules(template_id, label, delimiters)
            
            logger.debug(f"Updated persistent template {template_id} for label '{label}'")
        except Exception as e:
            logger.warning(f"Failed to update persistent template: {str(e)}")
    
    return template


def get_template(template_hash: str) -> Optional[Dict[str, Any]]:
    """Get a stored template by hash.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        Template dictionary if found, None otherwise.
    """
    return _templates.get(template_hash)


def get_template_field_positions(template_hash: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Get field positions (gabarito) for a template.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        Dictionary mapping field names to positions, or None if template not found.
    """
    template = _templates.get(template_hash)
    if template:
        return template.get("field_positions")
    return None


def get_template_alternative_positions(template_hash: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Get alternative positions for fields in a template.
    
    These are fallback positions learned when format validation fails.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        Dictionary mapping field names to lists of alternative positions, or None if template not found.
    """
    template = _templates.get(template_hash)
    if template:
        return template.get("alternative_positions")
    return None


def get_template_extraction_modes(template_hash: str) -> Optional[Dict[str, str]]:
    """Get extraction modes (exact/partial) for a template.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        Dictionary mapping field names to extraction modes ("exact" or "partial"),
        or None if template not found.
    """
    template = _templates.get(template_hash)
    if template:
        return template.get("extraction_modes")
    return None


def get_template_box_patterns(template_hash: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Get learned box patterns for a template.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        Dictionary mapping box text to pattern data, or None if template not found.
    """
    template = _templates.get(template_hash)
    if template:
        return template.get("box_patterns")
    return None


def get_template_box_delimiters(template_hash: str) -> Optional[Dict[str, List[str]]]:
    """Get learned delimiters for boxes in a template.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        Dictionary mapping box text to list of delimiters, or None if template not found.
    """
    template = _templates.get(template_hash)
    if template:
        return template.get("box_delimiters")
    return None


def get_template_extracted_data(template_hash: str) -> Optional[Dict[str, Any]]:
    """Get extracted data (ground truth) for a template.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        Dictionary of extracted field values, or None if template not found.
    """
    template = _templates.get(template_hash)
    if template:
        return template.get("extracted_data")
    return None


def has_template(template_hash: str) -> bool:
    """Check if a template exists.
    
    Args:
        template_hash: Perceptual hash of the template.
        
    Returns:
        True if template exists, False otherwise.
    """
    return template_hash in _templates


def list_templates() -> List[str]:
    """List all stored template hashes.
    
    Returns:
        List of template hash strings.
    """
    return list(_templates.keys())


def clear_templates() -> None:
    """Clear all stored templates."""
    _templates.clear()
    logger.info("Cleared all templates")


def save_templates(file_path: str) -> None:
    """Save templates to a JSON file for persistence.
    
    Args:
        file_path: Path to the JSON file where templates will be saved.
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_templates, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(_templates)} templates to {file_path}")


def load_templates(file_path: str) -> None:
    """Load templates from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing templates.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Templates file not found: {file_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    
    _templates.update(loaded)
    logger.info(f"Loaded {len(loaded)} templates from {file_path}")
