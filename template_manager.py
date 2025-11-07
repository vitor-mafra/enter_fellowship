"""Template manager for persistent learning and knowledge accumulation."""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TEMPLATES_DIR = "templates"
TEMPLATES_FILE = os.path.join(TEMPLATES_DIR, "templates.json")

# In-memory cache
_templates_cache: Optional[Dict[str, Any]] = None


def _ensure_templates_dir() -> None:
    """Ensure templates directory exists."""
    Path(TEMPLATES_DIR).mkdir(parents=True, exist_ok=True)


def generate_template_id(label: str, field_names: List[str]) -> str:
    """Generate a stable template_id from label and field structure.
    
    Args:
        label: Document label/type.
        field_names: List of field names in the extraction schema.
        
    Returns:
        Stable template_id string.
    """
    # Sort field names for consistency
    sorted_fields = sorted(field_names)
    fields_str = "|".join(sorted_fields)
    
    # Create hash from label + field structure
    combined = f"{label}:{fields_str}"
    hash_obj = hashlib.md5(combined.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()[:8]
    
    # Format: label_XXX where XXX is short hash
    template_id = f"{label}_{hash_hex}"
    
    return template_id


def infer_type(value: Any) -> str:
    """Infer the type of a field value.
    
    Args:
        value: Field value (can be string, number, None, etc.).
        
    Returns:
        Type string: "date", "time", "money", "number", or "text".
    """
    if value is None:
        return "text"
    
    value_str = str(value).strip()
    
    if not value_str:
        return "text"
    
    # Date pattern: DD/MM/YYYY or DD-MM-YYYY (must be exact match)
    date_pattern = re.compile(r'^\d{2}[/-]\d{2}[/-]\d{4}$')
    if date_pattern.match(value_str):
        return "date"
    
    # Time pattern: HH:MM or H:MM (must be exact match)
    time_pattern = re.compile(r'^\d{1,2}:\d{2}$')
    if time_pattern.match(value_str):
        return "time"
    
    # First, check if it contains letters (excluding currency symbols)
    # This catches names, addresses, etc. early
    # Create a version without currency symbols to check for letters
    temp_for_letter_check = re.sub(
        r'R\s*\$|R\$|\$|USD|EUR|€|£|BRL', 
        '', 
        value_str, 
        flags=re.IGNORECASE
    )
    
    # If there are letters (even after removing currency), it's text
    if re.search(r'[a-záàâãéêíóôõúç]', temp_for_letter_check, re.IGNORECASE):
        return "text"
    
    # Money pattern: must have currency symbol AND be primarily numeric
    # Currency symbols: R$, $, USD, EUR, €, £, BRL
    currency_pattern = re.compile(
        r'(R\s*\$|R\$|\$|USD|EUR|€|£|BRL)', 
        re.IGNORECASE
    )
    has_currency = bool(currency_pattern.search(value_str))
    
    if has_currency:
        # Remove currency symbols, spaces, and separators
        cleaned = re.sub(
            r'R\s*\$|R\$|\$|USD|EUR|€|£|BRL|\s', 
            '', 
            value_str, 
            flags=re.IGNORECASE
        )
        # Remove decimal separators (keep only digits)
        cleaned = re.sub(r'[.,]', '', cleaned)
        # Must be purely digits after cleaning
        if cleaned and re.match(r'^\d+$', cleaned) and len(cleaned) > 0:
            return "money"
    
    # Number pattern: purely numeric (with optional decimal separators)
    # Remove spaces first
    cleaned = re.sub(r'\s', '', value_str)
    
    # Pattern for numbers: digits with optional . or , as decimal separator
    # Can have thousands separators (. or ,) but must be valid number format
    number_patterns = [
        r'^\d+$',  # Pure integer: "123"
        r'^\d+[.,]\d+$',  # Decimal: "123.45" or "123,45"
        r'^\d{1,3}([.,]\d{3})+$',  # Thousands: "1.234.567" or "1,234,567"
        r'^\d{1,3}([.,]\d{3})*[.,]\d+$',  # Thousands with decimal: "1.234.567,89"
    ]
    
    for pattern in number_patterns:
        if re.match(pattern, cleaned):
            # Verify it has at least one digit
            if re.search(r'\d', cleaned):
                return "number"
    
    # Default to text
    return "text"


def update_avg_length(existing_avg: float, new_value: float, n: int) -> float:
    """Update average length incrementally.
    
    Args:
        existing_avg: Current average length.
        new_value: New value length to incorporate.
        n: Current number of samples (before adding new_value).
        
    Returns:
        Updated average length.
    """
    if n == 0:
        return new_value
    
    new_avg = (existing_avg * n + new_value) / (n + 1)
    return new_avg


def load_templates() -> Dict[str, Any]:
    """Load templates from JSON file.
    
    Returns:
        Dictionary with "templates" key containing list of template dicts.
    """
    global _templates_cache
    
    if _templates_cache is not None:
        return _templates_cache
    
    _ensure_templates_dir()
    
    if not os.path.exists(TEMPLATES_FILE):
        _templates_cache = {"templates": []}
        return _templates_cache
    
    try:
        # Check if file is empty
        if os.path.getsize(TEMPLATES_FILE) == 0:
            logger.debug("Templates file is empty, initializing empty")
            _templates_cache = {"templates": []}
            return _templates_cache
        
        with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # If file only contains whitespace, treat as empty
            if not content:
                logger.debug("Templates file contains only whitespace, initializing empty")
                _templates_cache = {"templates": []}
                return _templates_cache
            
            data = json.loads(content)
        
        if not isinstance(data, dict) or "templates" not in data:
            logger.warning("Invalid templates file structure, initializing empty")
            _templates_cache = {"templates": []}
            return _templates_cache
        
        _templates_cache = data
        logger.info(f"Loaded {len(data.get('templates', []))} templates from {TEMPLATES_FILE}")
        return _templates_cache
        
    except json.JSONDecodeError as e:
        logger.warning(f"Templates file contains invalid JSON: {str(e)}. Initializing empty.")
        _templates_cache = {"templates": []}
        return _templates_cache
    except IOError as e:
        logger.error(f"Failed to read templates file: {str(e)}")
        _templates_cache = {"templates": []}
        return _templates_cache


def save_templates(data: Dict[str, Any]) -> None:
    """Save templates to JSON file.
    
    Args:
        data: Dictionary with "templates" key containing list of template dicts.
    """
    global _templates_cache
    
    _ensure_templates_dir()
    
    try:
        with open(TEMPLATES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        _templates_cache = data
        logger.info(f"Saved {len(data.get('templates', []))} templates to {TEMPLATES_FILE}")
        
    except IOError as e:
        logger.error(f"Failed to save templates: {str(e)}")


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Get a template by template_id.
    
    Args:
        template_id: Template identifier.
        
    Returns:
        Template dictionary if found, None otherwise.
    """
    data = load_templates()
    templates = data.get("templates", [])
    
    for template in templates:
        if template.get("template_id") == template_id:
            return template
    
    return None


def find_template_by_label_and_fields(label: str, field_names: List[str]) -> Optional[Dict[str, Any]]:
    """Find template by label and field structure.
    
    Args:
        label: Document label.
        field_names: List of field names.
        
    Returns:
        Template dictionary if found, None otherwise.
    """
    template_id = generate_template_id(label, field_names)
    return get_template(template_id)


def update_template(
    template_id: str,
    label: str,
    extracted_fields: Dict[str, Any],
    extraction_modes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Update or create a template with new extracted field data.
    
    Args:
        template_id: Template identifier.
        label: Document label/type.
        extracted_fields: Dictionary of field_name -> field_value from extraction.
        
    Returns:
        Updated template dictionary.
    """
    data = load_templates()
    templates = data.get("templates", [])
    
    # Find existing template
    existing_template = None
    existing_index = -1
    for idx, template in enumerate(templates):
        if template.get("template_id") == template_id:
            existing_template = template
            existing_index = idx
            break
    
    if existing_template:
        # Update existing template
        fields = existing_template.get("fields", {})
        field_counts = existing_template.get("field_counts", {})
        
        for field_name, field_value in extracted_fields.items():
            if field_value is None:
                continue
            
            value_str = str(field_value).strip()
            if not value_str:
                continue
            
            # Infer type
            inferred_type = infer_type(field_value)
            value_length = len(value_str)
            
            if field_name in fields:
                # Update existing field
                existing_field = fields[field_name]
                existing_type = existing_field.get("type", "text")
                existing_avg = existing_field.get("avg_length", 0.0)
                n = field_counts.get(field_name, 0)
                
                # Update average length
                new_avg = update_avg_length(existing_avg, value_length, n)
                fields[field_name]["avg_length"] = new_avg
                
                # Update type (track frequency, use most common)
                type_history = existing_field.get("type_history", {})
                type_history[inferred_type] = type_history.get(inferred_type, 0) + 1
                fields[field_name]["type_history"] = type_history
                
                # Use most frequent type
                most_common_type = max(type_history.items(), key=lambda x: x[1])[0]
                fields[field_name]["type"] = most_common_type
                
                # Update extraction mode if provided
                if extraction_modes and field_name in extraction_modes:
                    mode = extraction_modes[field_name]
                    # Track mode frequency
                    mode_history = existing_field.get("mode_history", {})
                    mode_history[mode] = mode_history.get(mode, 0) + 1
                    fields[field_name]["mode_history"] = mode_history
                    # Use most frequent mode
                    most_common_mode = max(mode_history.items(), key=lambda x: x[1])[0]
                    fields[field_name]["extraction_mode"] = most_common_mode
                
                # Update count
                field_counts[field_name] = n + 1
            else:
                # New field
                field_data = {
                    "type": inferred_type,
                    "avg_length": float(value_length),
                    "type_history": {inferred_type: 1},
                }
                # Add extraction mode if provided
                if extraction_modes and field_name in extraction_modes:
                    mode = extraction_modes[field_name]
                    field_data["extraction_mode"] = mode
                    field_data["mode_history"] = {mode: 1}
                
                fields[field_name] = field_data
                field_counts[field_name] = 1
        
        existing_template["fields"] = fields
        existing_template["field_counts"] = field_counts
        templates[existing_index] = existing_template
        
        logger.info(
            f"Updated template {template_id} with {len(extracted_fields)} fields "
            f"({len(fields)} total fields)"
        )
    else:
        # Create new template
        fields = {}
        field_counts = {}
        
        for field_name, field_value in extracted_fields.items():
            if field_value is None:
                continue
            
            value_str = str(field_value).strip()
            if not value_str:
                continue
            
            inferred_type = infer_type(field_value)
            value_length = len(value_str)
            
            field_data = {
                "type": inferred_type,
                "avg_length": float(value_length),
                "type_history": {inferred_type: 1},
            }
            # Add extraction mode if provided
            if extraction_modes and field_name in extraction_modes:
                mode = extraction_modes[field_name]
                field_data["extraction_mode"] = mode
                field_data["mode_history"] = {mode: 1}
            
            fields[field_name] = field_data
            field_counts[field_name] = 1
        
        new_template = {
            "template_id": template_id,
            "label": label,
            "fields": fields,
            "field_counts": field_counts,
        }
        
        templates.append(new_template)
        logger.info(
            f"Created new template {template_id} for label '{label}' "
            f"with {len(fields)} fields"
        )
    
    data["templates"] = templates
    save_templates(data)
    
    return templates[existing_index] if existing_index >= 0 else new_template


def get_template_for_extraction(label: str, field_names: List[str]) -> Optional[Dict[str, Any]]:
    """Get or create template_id for a given label and field structure.
    
    Args:
        label: Document label.
        field_names: List of field names in extraction schema.
        
    Returns:
        Template dictionary if found, None if new template needed.
    """
    template_id = _generate_template_id(label, field_names)
    return get_template(template_id)


def learn_box_pattern(
    template_id: str,
    label: str,
    pattern_sequence: List[str],
) -> None:
    """Learn and store a box pattern (sequence of types) for a template.
    
    Args:
        template_id: Template identifier.
        label: Document label/type.
        pattern_sequence: List of inferred types (e.g., ["number", "uf", "text", "text"]).
    """
    data = load_templates()
    templates = data.get("templates", [])
    
    # Find template
    template = None
    template_index = -1
    for idx, t in enumerate(templates):
        if t.get("template_id") == template_id:
            template = t
            template_index = idx
            break
    
    if not template:
        logger.warning(f"Template {template_id} not found, cannot learn pattern")
        return
    
    # Initialize patterns if not exists
    if "patterns" not in template:
        template["patterns"] = []
    
    # Check if this pattern already exists
    pattern_str = ",".join(pattern_sequence)
    existing_pattern = None
    for p in template["patterns"]:
        if p.get("sequence") == pattern_sequence:
            existing_pattern = p
            break
    
    if existing_pattern:
        # Update frequency
        existing_pattern["frequency"] = existing_pattern.get("frequency", 1) + 1
        logger.debug(f"Updated pattern frequency for {template_id}: {pattern_sequence}")
    else:
        # Add new pattern
        template["patterns"].append({
            "label": label,
            "sequence": pattern_sequence,
            "frequency": 1,
        })
        logger.debug(f"Learned new pattern for {template_id}: {pattern_sequence}")
    
    templates[template_index] = template
    data["templates"] = templates
    save_templates(data)


def learn_split_rules(
    template_id: str,
    label: str,
    delimiters: List[str],
) -> None:
    """Learn and store split rules (delimiters) for a template.
    
    Args:
        template_id: Template identifier.
        label: Document label/type.
        delimiters: List of delimiter regex patterns that were effective.
    """
    data = load_templates()
    templates = data.get("templates", [])
    
    # Find template
    template = None
    template_index = -1
    for idx, t in enumerate(templates):
        if t.get("template_id") == template_id:
            template = t
            template_index = idx
            break
    
    if not template:
        logger.warning(f"Template {template_id} not found, cannot learn split rules")
        return
    
    # Initialize split_rules if not exists
    if "split_rules" not in template:
        template["split_rules"] = {
            "delimiters": [],
            "delimiter_frequencies": {},
        }
    
    split_rules = template["split_rules"]
    
    # Update delimiter frequencies
    for delimiter in delimiters:
        if delimiter in split_rules["delimiter_frequencies"]:
            split_rules["delimiter_frequencies"][delimiter] += 1
        else:
            split_rules["delimiter_frequencies"][delimiter] = 1
    
    # Update most frequent delimiters (top 5)
    sorted_delimiters = sorted(
        split_rules["delimiter_frequencies"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    split_rules["delimiters"] = [d[0] for d in sorted_delimiters[:5]]
    
    templates[template_index] = template
    data["templates"] = templates
    save_templates(data)
    
    logger.debug(
        f"Updated split rules for {template_id}: "
        f"top delimiters = {split_rules['delimiters']}"
    )


def clear_cache() -> None:
    """Clear the in-memory templates cache."""
    global _templates_cache
    _templates_cache = None
    logger.debug("Cleared templates cache")

