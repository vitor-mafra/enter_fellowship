"""Cache module for extraction results to avoid re-processing documents."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Modules that affect extraction results - if any of these change, cache is invalid
EXTRACTION_MODULES = [
    "pipeline.py",
    "heuristics.py",
    "oracle.py",
    "box_parser.py",
    "extraction.py",
    "hashing.py",
    "template_manager.py",
    "classification.py",
    "cache.py",  # Cache itself affects behavior
]


def _compute_file_hash(file_path: str) -> Optional[str]:
    """Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Hex digest of file hash, or None if file doesn't exist.
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return None
        
        hash_obj = hashlib.sha256()
        with open(file_path_obj, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.debug(f"Failed to compute hash for {file_path}: {str(e)}")
        return None


def _get_file_mtime(file_path: str) -> Optional[float]:
    """Get file modification time.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Modification time as float, or None if file doesn't exist.
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return None
        return file_path_obj.stat().st_mtime
    except Exception as e:
        logger.debug(f"Failed to get mtime for {file_path}: {str(e)}")
        return None


def _compute_code_version() -> str:
    """Compute a version hash of extraction code modules.
    
    Returns:
        Hex digest of combined module hashes.
    """
    hash_obj = hashlib.sha256()
    
    for module_name in EXTRACTION_MODULES:
        module_path = Path(module_name)
        if module_path.exists():
            module_hash = _compute_file_hash(module_name)
            if module_hash:
                hash_obj.update(module_hash.encode())
        else:
            # Module not found - include name to detect missing modules
            hash_obj.update(module_name.encode())
    
    return hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity


def _get_cached_code_version(result_file: str) -> Optional[str]:
    """Get code version stored in cached result metadata.
    
    Args:
        result_file: Path to cached result file.
        
    Returns:
        Code version string if found, None otherwise.
    """
    # Store code version in a separate metadata file
    metadata_file = Path(result_file).with_suffix(".meta.json")
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get("code_version")
    except Exception:
        return None


def _save_code_version(result_file: str, code_version: str) -> None:
    """Save code version to metadata file.
    
    Args:
        result_file: Path to result file.
        code_version: Code version string to save.
    """
    metadata_file = Path(result_file).with_suffix(".meta.json")
    
    try:
        metadata = {
            "code_version": code_version,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.debug(f"Failed to save code version metadata: {str(e)}")


def _schema_matches(cached_schema: Dict[str, str], current_schema: Dict[str, str]) -> bool:
    """Check if cached extraction schema matches current schema.
    
    Args:
        cached_schema: Schema stored in cache.
        current_schema: Current extraction schema.
        
    Returns:
        True if schemas match (same field names), False otherwise.
    """
    # Only check field names, not descriptions (descriptions may vary)
    cached_fields = set(cached_schema.keys())
    current_fields = set(current_schema.keys())
    return cached_fields == current_fields


def get_cached_result(
    pdf_path: str,
    extraction_schema: Dict[str, str],
    output_file: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get cached extraction result if available and valid.
    
    Checks:
    1. Result file exists
    2. PDF file hasn't changed (by modification time)
    3. Extraction schema matches (same field names)
    
    Args:
        pdf_path: Path to the PDF file.
        extraction_schema: Current extraction schema.
        output_file: Optional path to output file. If None, generates from PDF name.
        
    Returns:
        Cached extracted data if valid, None otherwise.
    """
    # Generate output filename if not provided
    if output_file is None:
        pdf_name = Path(pdf_path).stem
        output_file = f"results/{pdf_name}.json"
    
    result_path = Path(output_file)
    pdf_path_obj = Path(pdf_path)
    
    # Check if result file exists
    if not result_path.exists():
        logger.debug(f"Cache miss: result file not found: {output_file}")
        return None
    
    # Check if PDF file exists
    if not pdf_path_obj.exists():
        logger.warning(f"PDF file not found: {pdf_path}, but cache exists")
        return None
    
    # Check if PDF was modified after result was created
    pdf_mtime = _get_file_mtime(pdf_path)
    result_mtime = _get_file_mtime(str(result_path))
    
    if pdf_mtime is None or result_mtime is None:
        logger.debug(f"Cache miss: could not get file timestamps")
        return None
    
    # If PDF was modified after result was created, cache is invalid
    if pdf_mtime > result_mtime:
        logger.info(
            f"Cache invalid: PDF was modified after result was created "
            f"(PDF: {pdf_mtime}, Result: {result_mtime})"
        )
        return None
    
    # Check code version - if code changed, cache is invalid
    current_code_version = _compute_code_version()
    cached_code_version = _get_cached_code_version(str(result_path))
    
    if cached_code_version is None:
        # Old cache without code version - invalidate to be safe
        logger.info(
            f"Cache invalid: no code version metadata found (old cache format)"
        )
        return None
    
    if current_code_version != cached_code_version:
        logger.info(
            f"Cache invalid: extraction code changed "
            f"(cached: {cached_code_version}, current: {current_code_version})"
        )
        return None
    
    # Load cached result
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cached result from {output_file}: {str(e)}")
        return None
    
    # Verify that cached data has all required fields from schema
    required_fields = set(extraction_schema.keys())
    cached_fields = set(cached_data.keys())
    
    if not required_fields.issubset(cached_fields):
        missing_fields = required_fields - cached_fields
        logger.info(
            f"Cache invalid: cached result missing fields: {missing_fields}"
        )
        return None
    
    # Check if schema matches (optional - stored in metadata if available)
    # For now, we just verify all required fields are present
    
    logger.info(
        f"Cache hit: using cached result for {Path(pdf_path).name} "
        f"(PDF modified: {pdf_mtime}, Result created: {result_mtime}, Code version: {current_code_version})"
    )
    
    # Return only the fields from current extraction_schema
    result = {field: cached_data.get(field) for field in extraction_schema.keys()}
    
    return result


def save_to_cache(
    extracted_data: Dict[str, Any],
    output_file: str,
) -> None:
    """Save extraction result to cache with code version metadata.
    
    Args:
        extracted_data: Extracted data to cache.
        output_file: Path to output file where result will be saved.
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save extraction result
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        # Save code version metadata
        code_version = _compute_code_version()
        _save_code_version(str(output_path), code_version)
        
        logger.debug(f"Saved result to cache: {output_file} (code version: {code_version})")
    except Exception as e:
        logger.warning(f"Failed to save result to cache {output_file}: {str(e)}")


def clear_cache(result_file: Optional[str] = None) -> None:
    """Clear cache for a specific file or all files.
    
    Args:
        result_file: Optional path to specific result file. If None, clears all metadata files.
    """
    if result_file:
        # Clear specific file's metadata
        metadata_file = Path(result_file).with_suffix(".meta.json")
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"Cleared cache metadata for {result_file}")
    else:
        # Clear all metadata files in results directory
        results_dir = Path("results")
        if results_dir.exists():
            metadata_files = list(results_dir.glob("*.meta.json"))
            for metadata_file in metadata_files:
                metadata_file.unlink()
            logger.info(f"Cleared {len(metadata_files)} cache metadata files")

