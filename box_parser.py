"""Box parser for handling multi-field OCR text blocks."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from template_manager import infer_type

logger = logging.getLogger(__name__)

# Common delimiters for splitting text blocks
DEFAULT_DELIMITERS = [
    r'\s+-\s+',  # " - " (space-dash-space)
    r'\s+\|\s+',  # " | " (space-pipe-space)
    r'\s*:\s+',  # ": " (colon with optional space before)
    r'\s+/\s+',  # " / " (space-slash-space)
    r'\s+•\s+',  # " • " (bullet)
    r'\s*,\s+',  # ", " (comma)
    r'\s+',  # Multiple spaces (fallback)
]

# Weights for segment-to-field matching
W_TYPE_MATCH = 0.35
W_LENGTH_SIMILARITY = 0.25
W_POSITIONAL_PROXIMITY = 0.20
W_SCHEMA_KEYWORD_MATCH = 0.15
W_LABEL_PENALTY = 0.05


def segment_box_text(text: str, delimiters: Optional[List[str]] = None) -> List[str]:
    """Segment a text block into candidate subfields.
    
    Uses lexical and symbol-based splits, token-type transitions, and visual patterns.
    
    Args:
        text: Text content of the bounding box.
        delimiters: Optional list of regex patterns for splitting. Uses DEFAULT_DELIMITERS if None.
        
    Returns:
        List of segmented text pieces.
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    if delimiters is None:
        delimiters = DEFAULT_DELIMITERS
    
    # Try splitting with delimiters (in order of specificity)
    segments = [text]
    
    for delimiter in delimiters:
        new_segments = []
        for segment in segments:
            parts = re.split(delimiter, segment)
            new_segments.extend([p.strip() for p in parts if p.strip()])
        segments = new_segments
        if len(segments) > 1:
            break  # Use first delimiter that produces multiple segments
    
    # If no delimiter worked, try token-type transitions
    if len(segments) == 1:
        segments = _split_by_token_transitions(text)
    
    # Clean up segments
    segments = [s.strip() for s in segments if s.strip()]
    
    return segments


def _split_by_token_transitions(text: str) -> List[str]:
    """Split text by token type transitions (number → uppercase → lowercase, etc.).
    
    Args:
        text: Text to split.
        
    Returns:
        List of segments.
    """
    if not text:
        return []
    
    segments = []
    current_segment = []
    current_type = None
    
    words = text.split()
    
    for word in words:
        word_type = _get_word_type(word)
        
        if current_type is None:
            current_type = word_type
            current_segment.append(word)
        elif word_type == current_type:
            current_segment.append(word)
        else:
            # Type transition - split here
            if current_segment:
                segments.append(" ".join(current_segment))
            current_segment = [word]
            current_type = word_type
    
    if current_segment:
        segments.append(" ".join(current_segment))
    
    return segments if len(segments) > 1 else [text]


def _get_word_type(word: str) -> str:
    """Classify word type for transition detection.
    
    Args:
        word: Word to classify.
        
    Returns:
        Type: "number", "uppercase", "mixed", "lowercase", "other"
    """
    if re.match(r'^\d+$', word):
        return "number"
    elif word.isupper() and len(word) >= 2:
        return "uppercase"
    elif word.islower():
        return "lowercase"
    elif word[0].isupper():
        return "mixed"
    else:
        return "other"


def infer_segment_type(segment: str) -> str:
    """Infer the type of a text segment.
    
    Args:
        segment: Text segment to analyze.
        
    Returns:
        Type string: "number", "uf", "date", "time", "money", "text", etc.
    """
    if not segment:
        return "text"
    
    segment = segment.strip()
    
    # Use template_manager's infer_type for basic types
    basic_type = infer_type(segment)
    
    # Additional heuristics for specific types
    if basic_type == "text":
        # Check for UF (2-letter uppercase Brazilian state codes)
        if len(segment) == 2 and segment.isupper() and segment.isalpha():
            # Common Brazilian state codes
            uf_codes = {
                "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
                "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
                "RS", "RO", "RR", "SC", "SP", "SE", "TO"
            }
            if segment in uf_codes:
                return "uf"
        
        # Check for common abbreviations
        if len(segment) <= 4 and segment.isupper():
            return "abbreviation"
    
    return basic_type


def learn_box_pattern(segments: List[str]) -> List[str]:
    """Learn the type pattern from a list of segments.
    
    Args:
        segments: List of text segments from a box.
        
    Returns:
        List of inferred types for each segment.
    """
    return [infer_segment_type(seg) for seg in segments]


def _compute_type_match_score(segment_type: str, expected_type: Optional[str]) -> float:
    """Compute type match score.
    
    Args:
        segment_type: Inferred type of the segment.
        expected_type: Expected type from template.
        
    Returns:
        Score between 0.0 and 1.0.
    """
    if expected_type is None:
        return 0.5
    
    if segment_type == expected_type:
        return 1.0
    
    # Partial matches
    if (segment_type == "abbreviation" and expected_type == "text") or \
       (expected_type == "abbreviation" and segment_type == "text"):
        return 0.7
    
    if (segment_type == "uf" and expected_type == "text") or \
       (expected_type == "uf" and segment_type == "text"):
        return 0.6
    
    return 0.0


def _compute_length_similarity_score(
    segment_length: int,
    expected_avg_length: Optional[float],
) -> float:
    """Compute length similarity score.
    
    Args:
        segment_length: Length of the segment.
        expected_avg_length: Expected average length from template.
        
    Returns:
        Score between 0.0 and 1.0.
    """
    if expected_avg_length is None or expected_avg_length == 0:
        return 0.5
    
    diff = abs(segment_length - expected_avg_length)
    relative_diff = diff / max(expected_avg_length, 1.0)
    
    if relative_diff <= 0.1:
        return 1.0
    elif relative_diff <= 0.3:
        return 0.8
    elif relative_diff <= 0.5:
        return 0.6
    elif relative_diff <= 1.0:
        return 0.4
    else:
        return max(0.0, 0.2 - (relative_diff - 1.0) * 0.2)


def _compute_positional_proximity_score(
    segment_index: int,
    expected_position: Optional[int],
    total_segments: int,
) -> float:
    """Compute positional proximity score.
    
    Args:
        segment_index: Index of the segment in the box (0-based).
        expected_position: Expected position index from learned pattern (None if unknown).
        total_segments: Total number of segments in the box.
        
    Returns:
        Score between 0.0 and 1.0.
    """
    if expected_position is None:
        return 0.5
    
    # Normalize positions to [0, 1]
    normalized_segment = segment_index / max(total_segments - 1, 1)
    normalized_expected = expected_position / max(total_segments - 1, 1)
    
    diff = abs(normalized_segment - normalized_expected)
    
    if diff <= 0.1:
        return 1.0
    elif diff <= 0.2:
        return 0.8
    elif diff <= 0.3:
        return 0.6
    else:
        return max(0.0, 0.4 - diff * 0.5)


def _compute_schema_keyword_match_score(
    segment: str,
    schema_description: str,
) -> float:
    """Compute semantic similarity between segment and schema description.
    
    Args:
        segment: Text segment.
        schema_description: Field description from extraction_schema.
        
    Returns:
        Score between 0.0 and 1.0.
    """
    if not segment or not schema_description:
        return 0.0
    
    segment_lower = segment.lower()
    desc_lower = schema_description.lower()
    
    # Extract keywords from description (words longer than 3 chars)
    desc_words = [w for w in re.findall(r'\b\w{4,}\b', desc_lower)]
    segment_words = [w for w in re.findall(r'\b\w{4,}\b', segment_lower)]
    
    if not desc_words or not segment_words:
        return 0.0
    
    # Count matches
    matches = sum(1 for word in desc_words if word in segment_words)
    match_ratio = matches / max(len(desc_words), 1)
    
    return min(1.0, match_ratio * 2.0)  # Scale up for better scores


def _compute_label_penalty(segment: str) -> float:
    """Compute penalty for label-like text (e.g., "Nome:", "Data:").
    
    Args:
        segment: Text segment to check.
        
    Returns:
        Penalty score (higher = more likely to be a label, should be subtracted).
    """
    segment_lower = segment.lower().strip()
    
    # Check for colon at end
    if segment_lower.endswith(':'):
        return 0.8
    
    # Check for common label patterns
    label_patterns = [
        r'^[a-záàâãéêíóôõúç\s]+:\s*$',
        r'^[a-záàâãéêíóôõúç\s]+-\s*$',
    ]
    
    for pattern in label_patterns:
        if re.match(pattern, segment_lower):
            return 0.6
    
    # Check if it's a very short word (likely a label)
    if len(segment_lower) <= 3 and segment_lower.isalpha():
        return 0.3
    
    return 0.0


def match_segments_to_schema(
    segments: List[str],
    extraction_schema: Dict[str, str],
    template_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Match segmented pieces to expected fields in extraction_schema.
    
    Args:
        segments: List of text segments from a box.
        extraction_schema: Dictionary mapping field names to descriptions.
        template_data: Optional template data with learned patterns and field metadata.
        
    Returns:
        Dictionary mapping field names to matched segment values (or None if no match).
    """
    if not segments:
        return {field: None for field in extraction_schema.keys()}
    
    # Get learned pattern if available
    learned_pattern = None
    if template_data:
        patterns = template_data.get("patterns", [])
        if patterns:
            # Use the first pattern (could be enhanced to match by label)
            learned_pattern = patterns[0].get("sequence", [])
    
    # Infer types for all segments
    segment_types = [infer_segment_type(seg) for seg in segments]
    
    # Get field metadata from template
    field_metadata = {}
    if template_data:
        fields = template_data.get("fields", {})
        field_metadata = fields
    
    # Score each segment-field pair
    field_scores: Dict[str, List[Tuple[int, float]]] = {
        field: [] for field in extraction_schema.keys()
    }
    
    for field_name, field_desc in extraction_schema.items():
        expected_type = None
        expected_avg_length = None
        expected_position = None
        
        if field_name in field_metadata:
            metadata = field_metadata[field_name]
            expected_type = metadata.get("type")
            expected_avg_length = metadata.get("avg_length")
        
        # Find expected position from learned pattern
        if learned_pattern and expected_type:
            try:
                expected_position = learned_pattern.index(expected_type)
            except ValueError:
                pass
        
        for seg_idx, segment in enumerate(segments):
            segment_type = segment_types[seg_idx]
            segment_length = len(segment)
            
            # Compute component scores
            type_score = _compute_type_match_score(segment_type, expected_type)
            length_score = _compute_length_similarity_score(segment_length, expected_avg_length)
            position_score = _compute_positional_proximity_score(
                seg_idx, expected_position, len(segments)
            )
            keyword_score = _compute_schema_keyword_match_score(segment, field_desc)
            label_penalty = _compute_label_penalty(segment)
            
            # Composite score
            composite_score = (
                W_TYPE_MATCH * type_score +
                W_LENGTH_SIMILARITY * length_score +
                W_POSITIONAL_PROXIMITY * position_score +
                W_SCHEMA_KEYWORD_MATCH * keyword_score -
                W_LABEL_PENALTY * label_penalty
            )
            
            field_scores[field_name].append((seg_idx, composite_score))
    
    # Assign segments to fields (greedy: highest score wins, no double assignment)
    assignments: Dict[str, Optional[str]] = {field: None for field in extraction_schema.keys()}
    used_segments = set()
    
    # Sort all field-segment pairs by score (descending)
    all_pairs = []
    for field_name, scores in field_scores.items():
        for seg_idx, score in scores:
            all_pairs.append((field_name, seg_idx, score))
    
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Assign greedily
    for field_name, seg_idx, score in all_pairs:
        if score < 0.3:  # Minimum threshold
            continue
        
        if field_name not in assignments or assignments[field_name] is None:
            if seg_idx not in used_segments:
                assignments[field_name] = segments[seg_idx]
                used_segments.add(seg_idx)
                logger.debug(
                    f"Matched segment {seg_idx} '{segments[seg_idx]}' to field '{field_name}' "
                    f"(score: {score:.3f})"
                )
    
    return assignments


def extract_fields_sequential(
    box_text: str,
    extraction_schema: Dict[str, str],
    template_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Extract fields sequentially using heuristic: number first, then fixed-size string, then rest.
    
    This function extracts fields in order:
    1. First extracts numeric values
    2. Then extracts fixed-size strings (typically 2 chars for UF)
    3. Then extracts the remaining text
    
    Each extracted value is removed from the remaining text to avoid re-extraction.
    
    Args:
        box_text: Full text content of the bounding box.
        extraction_schema: Dictionary mapping field names to descriptions.
        template_data: Optional template data with field metadata.
        
    Returns:
        Dictionary mapping field names to extracted values (or None if no match).
    """
    if not box_text or not box_text.strip():
        return {field: None for field in extraction_schema.keys()}

    ordered_fields = list(extraction_schema.keys())

    if len(ordered_fields) > 1:
        positional_assignments = _extract_by_positional_split(box_text, ordered_fields)
        if positional_assignments is not None:
            logger.debug(
                "Positional split assignments for '%s': %s",
                box_text[:50],
                positional_assignments,
            )
            return positional_assignments

    assignments: Dict[str, Optional[str]] = {field: None for field in ordered_fields}
    remaining_text = box_text.strip()
    
    # Get field metadata from template
    field_metadata = {}
    if template_data:
        fields = template_data.get("fields", {})
        field_metadata = fields
    
    # Sort fields by priority: number first, then fixed-size, then rest
    field_priority = []
    for field_name in ordered_fields:
        metadata = field_metadata.get(field_name, {})
        expected_type = metadata.get("type", "text")
        expected_avg_length = metadata.get("avg_length")
        
        # Try to infer type from field name if not in metadata
        if not expected_type or expected_type == "text":
            field_lower = field_name.lower()
            if "numero" in field_lower or "num" in field_lower or "inscricao" in field_lower or "codigo" in field_lower:
                # Check if it's likely a number field
                if not expected_avg_length or expected_avg_length < 10:
                    expected_type = "number"
            elif "uf" in field_lower or "estado" in field_lower or "seccional" in field_lower:
                # Likely a 2-character state code
                if not expected_avg_length:
                    expected_avg_length = 2.0
        
        # Priority: 0 = number, 1 = fixed-size (2-4 chars), 2 = rest
        if expected_type == "number":
            priority = 0
        elif expected_avg_length and 2 <= expected_avg_length <= 4:
            priority = 1
        else:
            priority = 2
        
        field_priority.append((priority, expected_avg_length or 0, field_name))
    
    # Sort by priority (lower number = higher priority)
    field_priority.sort(key=lambda x: (x[0], -x[1]))  # Sort by priority, then by length (descending)
    
    logger.debug(
        f"Extracting fields sequentially from '{box_text}': "
        f"order = {[f[2] for f in field_priority]}"
    )
    
    # Extract fields in priority order
    for priority, expected_length, field_name in field_priority:
        if not remaining_text.strip():
            break
        
        metadata = field_metadata.get(field_name, {})
        expected_type = metadata.get("type", "text")
        expected_avg_length = metadata.get("avg_length")
        
        value = None
        
        # Step 1: Try to extract number first
        if expected_type == "number":
            import re
            # Find first number sequence in remaining text
            number_match = re.search(r'\d+', remaining_text)
            if number_match:
                value = number_match.group(0)
                # Remove extracted number and surrounding spaces/delimiters from remaining text
                start = number_match.start()
                end = number_match.end()
                # Remove the number and any spaces/delimiters around it
                # Get text before and after
                before = remaining_text[:start].rstrip()
                after = remaining_text[end:].lstrip()
                # Remove any leading/trailing spaces and delimiters from after
                # Remove common delimiters: space, dash, pipe, etc.
                after = re.sub(r'^[\s\-|]+', '', after)
                # Reconstruct remaining text
                if before and after:
                    remaining_text = before + " " + after
                else:
                    remaining_text = before + after
                remaining_text = remaining_text.strip()
                logger.debug(
                    f"Extracted number '{value}' for field '{field_name}', "
                    f"remaining: '{remaining_text[:50]}...'"
                )
        
        # Step 2: Try to extract fixed-size string (typically 2 chars for UF)
        elif expected_avg_length and 2 <= expected_avg_length <= 4 and not value:
            # Try to find a word/token of the expected length
            import re
            # Look for word boundaries with expected length (case-insensitive)
            pattern = r'\b\w{' + str(int(expected_avg_length)) + r'}\b'
            match = re.search(pattern, remaining_text, re.IGNORECASE)
            if match:
                candidate = match.group(0)
                # Prefer uppercase (likely UF or abbreviation)
                if candidate.isupper() or len(candidate) == int(expected_avg_length):
                    value = candidate
                    # Remove extracted value and surrounding spaces/delimiters from remaining text
                    start = match.start()
                    end = match.end()
                    before = remaining_text[:start].rstrip()
                    after = remaining_text[end:].lstrip()
                    # Remove any leading/trailing spaces and delimiters from after
                    after = re.sub(r'^[\s\-|]+', '', after)
                    # Reconstruct remaining text
                    if before and after:
                        remaining_text = before + " " + after
                    else:
                        remaining_text = before + after
                    remaining_text = remaining_text.strip()
                    logger.debug(
                        f"Extracted fixed-size '{value}' for field '{field_name}', "
                        f"remaining: '{remaining_text[:50]}...'"
                    )
        
        # Step 3: Extract the rest (remaining text)
        if not value and remaining_text.strip():
            # For the last field or if no specific pattern matched, take remaining text
            # But only if it's the last field or if expected_length suggests it's a longer text
            if priority == 2 or (expected_avg_length and expected_avg_length > 10):
                value = remaining_text.strip()
                remaining_text = ""
                logger.debug(
                    f"Extracted remaining text '{value[:50]}...' for field '{field_name}'"
                )
        
        if value:
            assignments[field_name] = value
    
    return assignments


def _extract_by_positional_split(
    box_text: str,
    field_order: List[str],
) -> Optional[Dict[str, Optional[str]]]:
    """Attempt extraction by splitting left-to-right using whitespace.

    Each field except the last receives the next token (as separated by
    whitespace). The last field receives the remaining text intact.
    Returns ``None`` if there are not enough tokens to safely distribute the
    values (i.e., fewer tokens than number of fields minus one).
    """
    if not field_order:
        return None

    text = (box_text or "").strip()
    if not text:
        return {field: None for field in field_order}

    required_tokens = max(len(field_order) - 1, 0)
    tokens = text.split()

    if len(tokens) < required_tokens:
        return None

    assignments: Dict[str, Optional[str]] = {}
    remaining = text

    for idx, field_name in enumerate(field_order):
        if idx < len(field_order) - 1:
            parts = remaining.strip().split(None, 1)
            if not parts:
                return None
            head = parts[0]
            tail = parts[1] if len(parts) > 1 else ""
            assignments[field_name] = head if head else None
            remaining = tail
        else:
            value = remaining.strip()
            assignments[field_name] = value if value else None

    return assignments


def extract_fields_from_box(
    box_text: str,
    extraction_schema: Dict[str, str],
    template_data: Optional[Dict[str, Any]] = None,
    delimiters: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """Complete pipeline: segment box text and match to schema fields.
    
    Uses sequential extraction heuristic when multiple fields need to be extracted
    from the same box (extraction_mode="partial").
    
    Args:
        box_text: Full text content of the bounding box.
        extraction_schema: Dictionary mapping field names to descriptions.
        template_data: Optional template data with learned patterns.
        delimiters: Optional custom delimiters for splitting.
        
    Returns:
        Dictionary mapping field names to extracted values (or None if no match).
    """
    if not box_text or not box_text.strip():
        return {field: None for field in extraction_schema.keys()}
    
    # If multiple fields, use sequential extraction heuristic
    if len(extraction_schema) > 1:
        logger.debug(
            f"Using sequential extraction for {len(extraction_schema)} fields from box: '{box_text[:50]}...'"
        )
        return extract_fields_sequential(box_text, extraction_schema, template_data)
    
    # For single field, use original segmentation approach
    # Get learned delimiters from template if available
    if template_data and delimiters is None:
        split_rules = template_data.get("split_rules", {})
        if split_rules:
            delimiters = split_rules.get("delimiters")
    
    # Segment the box text
    segments = segment_box_text(box_text, delimiters)
    
    if not segments:
        return {field: None for field in extraction_schema.keys()}
    
    logger.debug(
        f"Segmented box text '{box_text}' into {len(segments)} segments: {segments}"
    )
    
    # Match segments to schema fields
    assignments = match_segments_to_schema(segments, extraction_schema, template_data)
    
    return assignments

