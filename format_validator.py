"""Format validation utilities for extracted field values."""

import re
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def validate_date(value: Any) -> bool:
    """Validate if value matches a date format.
    
    Supports formats:
    - DD/MM/YYYY
    - DD-MM-YYYY
    - YYYY-MM-DD
    
    Args:
        value: Value to validate.
        
    Returns:
        True if value matches date format, False otherwise.
    """
    if value is None:
        return False
    
    value_str = str(value).strip()
    if not value_str:
        return False
    
    # Date patterns
    date_patterns = [
        r'^\d{2}[/-]\d{2}[/-]\d{4}$',  # DD/MM/YYYY or DD-MM-YYYY
        r'^\d{4}[/-]\d{2}[/-]\d{2}$',  # YYYY-MM-DD
    ]
    
    for pattern in date_patterns:
        if re.match(pattern, value_str):
            return True
    
    return False


def validate_time(value: Any) -> bool:
    """Validate if value matches a time format.
    
    Supports formats:
    - HH:MM
    - H:MM
    - HH:MM:SS
    
    Args:
        value: Value to validate.
        
    Returns:
        True if value matches time format, False otherwise.
    """
    if value is None:
        return False
    
    value_str = str(value).strip()
    if not value_str:
        return False
    
    # Time patterns
    time_patterns = [
        r'^\d{1,2}:\d{2}$',      # H:MM or HH:MM
        r'^\d{1,2}:\d{2}:\d{2}$',  # HH:MM:SS
    ]
    
    for pattern in time_patterns:
        if re.match(pattern, value_str):
            return True
    
    return False


def validate_number(value: Any) -> bool:
    """Validate if value is a number (integer or decimal).
    
    Args:
        value: Value to validate.
        
    Returns:
        True if value is a number, False otherwise.
    """
    if value is None:
        return False
    
    value_str = str(value).strip()
    if not value_str:
        return False
    
    # Explicit patterns for common document identifiers (CPF, CNPJ)
    cpf_patterns = [
        r'^\d{11}$',
        r'^\d{3}\.\d{3}\.\d{3}-\d{2}$',
    ]
    cnpj_patterns = [
        r'^\d{14}$',
        r'^\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}$',
    ]
    for pattern in cpf_patterns + cnpj_patterns:
        if re.match(pattern, value_str):
            return True
    
    # Remove common formatting (spaces, dots, commas, hyphens, slashes)
    # Keep leading minus sign for negative numbers if present
    cleaned = value_str.replace(' ', '').replace('.', '').replace(',', '')
    cleaned = cleaned.replace('/', '')
    if len(cleaned) > 1:
        cleaned = cleaned.replace('-', '')
    
    # Check if it's all digits (possibly with one decimal separator)
    if re.match(r'^\d+([.,]\d+)?$', value_str):
        return True
    
    # Also check if it's a pure integer
    try:
        int(cleaned)
        return True
    except ValueError:
        pass
    
    # Check for decimal numbers
    try:
        float(value_str.replace(',', '.'))
        return True
    except ValueError:
        pass
    
    return False


def validate_uf(value: Any) -> bool:
    """Validate if value matches a Brazilian UF (two-letter state code).

    Args:
        value: Value to validate.

    Returns:
        True if value consists of exactly two alphabetic characters.
    """
    if value is None:
        return False

    value_str = str(value).strip()
    if not value_str:
        return False

    return bool(re.match(r'^[A-Za-z]{2}$', value_str))


def infer_expected_format(field_name: str, field_description: Optional[str] = None) -> Optional[str]:
    """Infer expected format from field name and description.
    
    Args:
        field_name: Name of the field.
        field_description: Optional description of the field.
        
    Returns:
        Expected format: "date", "time", "number", or None if unknown.
    """
    field_lower = field_name.lower()
    desc_lower = (field_description or "").lower()
    
    combined = f"{field_lower} {desc_lower}"
    
    # UF/state indicators (must check before generic number detection)
    if re.search(r'\buf\b', combined):
        return "uf"

    # Date indicators
    date_keywords = ['data', 'date', 'nascimento', 'expedicao', 'vencimento', 'validade']
    if any(keyword in combined for keyword in date_keywords):
        return "date"
    
    # Time indicators
    time_keywords = ['hora', 'time', 'horario']
    if any(keyword in combined for keyword in time_keywords):
        return "time"
    
    # Number indicators
    number_keywords = ['numero', 'number', 'num', 'cpf', 'cnpj', 'registro', 'quantidade', 'valor']
    if any(keyword in combined for keyword in number_keywords):
        return "number"
    
    return None


def validate_format(value: Any, expected_format: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Validate if value matches expected format.
    
    Args:
        value: Value to validate.
        expected_format: Expected format ("date", "time", "number", or None).
        
    Returns:
        Tuple of (is_valid, detected_format).
        is_valid: True if value matches expected format.
        detected_format: Detected format ("date", "time", "number", or None).
    """
    if expected_format is None:
        return True, None
    
    # Detect actual format
    detected_format = None
    if validate_date(value):
        detected_format = "date"
    elif validate_time(value):
        detected_format = "time"
    elif validate_number(value):
        detected_format = "number"
    elif validate_uf(value):
        detected_format = "uf"
    
    # Check if detected format matches expected
    is_valid = (detected_format == expected_format) if detected_format else False
    
    return is_valid, detected_format


def extract_matching_value(value: Any, expected_format: Optional[str]) -> Optional[str]:
    """Extract a substring that matches the expected format from the value.

    Args:
        value: Original value extracted from document.
        expected_format: Expected format ("date", "time", "number", or None).

    Returns:
        Matching substring if found, otherwise None.
    """
    if value is None or expected_format is None:
        return None

    value_str = str(value)
    if not value_str:
        return None

    patterns: list[str] = []

    if expected_format == "date":
        patterns = [
            r"\d{2}[/-]\d{2}[/-]\d{4}",
            r"\d{4}[/-]\d{2}[/-]\d{2}",
        ]
    elif expected_format == "time":
        patterns = [
            r"\d{1,2}:\d{2}:\d{2}",
            r"\d{1,2}:\d{2}",
        ]
    elif expected_format == "number":
        patterns = [
            r"\d{3}\.\d{3}\.\d{3}-\d{2}",  # CPF masked
            r"\d{11}",
            r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",  # CNPJ masked
            r"\d{14}",
            r"\d+[.,]\d+",
            r"\d+",
        ]
    elif expected_format == "uf":
        patterns = [
            r"\b[A-Za-z]{2}\b",
        ]

    for pattern in patterns:
        match = re.search(pattern, value_str)
        if match:
            return match.group(0).upper()

    return None

