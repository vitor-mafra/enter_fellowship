"""Document classification module."""

import logging
from typing import Dict, Literal

from llm_utils import classify_document_structure

logger = logging.getLogger(__name__)

# Cache for document type classifications: label -> "stricted" | "flexible"
_classification_cache: Dict[str, Literal["stricted", "flexible"]] = {}


def classify_document_type(label: str) -> Literal["stricted", "flexible"]:
    """Classify a document as stricted or flexible based on its label.
    
    Uses an LLM to determine if a document has a stricted structure (fixed layout,
    stable field positions) or flexible structure (free-form text, variable layout).
    
    The classification is cached in memory - once a label is classified, the result
    is stored and reused for subsequent calls with the same label.
    
    Args:
        label: Document label/type identifier (e.g., "carteira_oab", "tela_sistema").
        
    Returns:
        Classification string: "stricted" or "flexible".
        
    Raises:
        ValueError: If classification fails.
    """
    # Check cache first
    if label in _classification_cache:
        logger.debug(f"Using cached classification for '{label}': {_classification_cache[label]}")
        return _classification_cache[label]
    
    logger.info(f"Classifying document type for label: {label}")
    try:
        result = classify_document_structure(label)
        logger.info(f"Document '{label}' classified as: {result}")
        
        # Store in cache
        _classification_cache[label] = result
        
        return result
    except Exception as e:
        logger.error(f"Failed to classify document type '{label}': {str(e)}")
        raise


def clear_classification_cache() -> None:
    """Clear the classification cache."""
    _classification_cache.clear()
    logger.info("Cleared classification cache")


def get_classification_cache() -> Dict[str, Literal["stricted", "flexible"]]:
    """Get the current classification cache.
    
    Returns:
        Dictionary mapping labels to their classifications.
    """
    return _classification_cache.copy()
