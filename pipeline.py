"""Document extraction pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cache import get_cached_result, save_to_cache
from classification import classify_document_type
from extraction import extract_text_simple, extract_text_with_coords
from format_validator import infer_expected_format, validate_format, extract_matching_value
from hashing import compare_phashes, compute_phash, compute_diff_hashes, find_similar_template as find_similar_template_multi_region
from heuristics import extract_fields_with_heuristics
from oracle import (
    create_template,
    extract_with_llm,
    get_template,
    get_template_alternative_positions,
    get_template_box_patterns,
    get_template_extraction_modes,
    get_template_field_positions,
    has_template,
    list_templates,
    _extract_field_positions,
)

logger = logging.getLogger(__name__)

# Output results accumulator
_extraction_results: List[Dict[str, Any]] = []

# Controls incremental combined results file writes
_incremental_enabled: bool = True

# Cache toggle (set by CLI)
_cache_enabled: bool = True

# Benchmark state (set by CLI)
_benchmark_enabled: bool = False
_benchmark_times: List[float] = []
_benchmark_first_time: Optional[float] = None
_benchmark_llm_fields_used: int = 0
_benchmark_total_fields: int = 0


def enable_benchmark(enabled: bool = True) -> None:
    """Enable or disable benchmark mode and reset metrics."""
    global _benchmark_enabled, _benchmark_times, _benchmark_first_time
    global _benchmark_llm_fields_used, _benchmark_total_fields
    _benchmark_enabled = enabled
    _benchmark_times = []
    _benchmark_first_time = None
    _benchmark_llm_fields_used = 0
    _benchmark_total_fields = 0


def get_benchmark_stats() -> Dict[str, Any]:
    """Return current benchmark statistics."""
    import statistics
    times = list(_benchmark_times)
    n = len(times)
    std_s = statistics.stdev(times) if n > 1 else 0.0
    return {
        "count": n,
        "total_s": sum(times) if n else 0.0,
        "min_s": min(times) if n else 0.0,
        "max_s": max(times) if n else 0.0,
        "median_s": statistics.median(times) if n else 0.0,
        "mean_s": (sum(times) / n) if n else 0.0,
        "std_s": std_s,
        "first_s": _benchmark_first_time if _benchmark_first_time is not None else 0.0,
        "llm_fields_used": _benchmark_llm_fields_used,
        "total_fields": _benchmark_total_fields,
    }


def get_benchmark_times() -> List[float]:
    """Return the raw per-file durations collected in this process."""
    return list(_benchmark_times)


def enable_incremental(enabled: bool = True) -> None:
    """Enable/disable incremental combined results writing within this process."""
    global _incremental_enabled
    _incremental_enabled = enabled


def enable_cache(enabled: bool = True) -> None:
    """Enable/disable cache usage (both read and write) within this process."""
    global _cache_enabled
    _cache_enabled = enabled


def _find_matching_template(pdf_path: str, label: Optional[str] = None, threshold: int = 40) -> Optional[str]:
    """Find a matching template hash for a PDF using multi-region hashing.
    
    Args:
        pdf_path: Path to the PDF file.
        label: Optional document label to filter templates.
        threshold: Legacy parameter (not used with multi-region hashing).
        
    Returns:
        Template hash if a match is found, None otherwise.
    """
    from oracle import get_template, list_templates
    
    # Build known templates dict from stored templates
    known_templates = {}
    for template_hash in list_templates():
        template = get_template(template_hash)
        if template:
            known_templates[template_hash] = {
                "label": template.get("label"),
                "hash_full": template.get("hash_full", template_hash),
                "hash_left": template.get("hash_left", ""),
                "hash_right": template.get("hash_right", ""),
            }
    
    # Use multi-region hashing to find similar template
    best_match = find_similar_template_multi_region(
        pdf_path,
        known_templates,
        label=label,
    )
    
    return best_match


def _extract_using_positions(
    text_blocks: List[Dict[str, Any]],
    template_hash: str,
    extraction_schema: Dict[str, str],
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract data using learned positional mappings from template.
    
    Uses heuristic-based matching to find the best text blocks for each field.
    For fields that fail positional extraction, uses incremental LLM extraction
    (only for missing fields, not re-extracting known fields).
    
    Args:
        text_blocks: Already extracted text blocks with coordinates.
        template_hash: Hash of the known template.
        extraction_schema: Dictionary mapping field names to extraction descriptions.
        label: Optional document label for loading template knowledge.
        
    Returns:
        Dictionary with extracted field values.
    """
    field_positions = get_template_field_positions(template_hash)
    alternative_positions = get_template_alternative_positions(template_hash)
    extraction_modes = get_template_extraction_modes(template_hash)
    box_patterns = get_template_box_patterns(template_hash)
    
    if not field_positions:
        logger.warning(
            f"Template {template_hash} has no stored positions, falling back to LLM"
        )
        return extract_with_llm(text_blocks, extraction_schema)
    
    # Heuristic-based extraction with primary positions
    logger.info(f"Using heuristic-based positional extraction for template {template_hash}")
    
    extracted_data = extract_fields_with_heuristics(
        field_positions,
        text_blocks,
        extraction_schema,
        label=label,
        extraction_modes=extraction_modes,
        box_patterns=box_patterns,
    )
    
    # Try alternative positions for fields that failed
    if alternative_positions:
        failed_fields = [
            field_name for field_name, value in extracted_data.items()
            if value is None
        ]
        
        for field_name in failed_fields:
            if field_name in alternative_positions:
                alt_positions = alternative_positions[field_name]
                logger.info(
                    f"Trying {len(alt_positions)} alternative positions for '{field_name}'"
                )
                
                # Try each alternative position
                for alt_pos in alt_positions:
                    # Create temporary field_positions dict with just this alternative
                    temp_positions = {field_name: alt_pos}
                    temp_extracted = extract_fields_with_heuristics(
                        temp_positions,
                        text_blocks,
                        {field_name: extraction_schema[field_name]},
                        label=label,
                        extraction_modes=extraction_modes,
                        box_patterns=box_patterns,
                    )
                    
                    if temp_extracted.get(field_name) is not None:
                        # Validate format before accepting alternative position
                        expected_format = infer_expected_format(
                            field_name,
                            extraction_schema.get(field_name)
                        )
                        
                        if expected_format:
                            is_valid, _ = validate_format(
                                temp_extracted[field_name],
                                expected_format
                            )
                            if not is_valid:
                                logger.debug(
                                    f"Alternative position for '{field_name}' "
                                    f"has invalid format, trying next alternative"
                                )
                                continue
                        
                        extracted_data[field_name] = temp_extracted[field_name]
                        logger.info(
                            f"Successfully extracted '{field_name}' using alternative position"
                        )
                        break
    
    # Validate extracted values against expected formats (after trying alternatives)
    format_invalid_fields = []
    for field_name, value in extracted_data.items():
        if value is None:
            continue
        
        # Infer expected format from field name/description
        expected_format = infer_expected_format(
            field_name,
            extraction_schema.get(field_name)
        )
        
        if expected_format:
            is_valid, detected_format = validate_format(value, expected_format)
            if not is_valid:
                # Try extracting a matching substring before flagging invalid
                extracted_value = extract_matching_value(value, expected_format)
                if extracted_value and extracted_value != value:
                    corrected_valid, _ = validate_format(extracted_value, expected_format)
                    if corrected_valid:
                        extracted_data[field_name] = extracted_value
                        logger.info(
                            f"Auto-corrected '{field_name}' using regex heuristic: '{value}' -> '{extracted_value}'"
                        )
                        continue
                
                logger.warning(
                    f"Format validation failed for '{field_name}': "
                    f"expected {expected_format}, got {detected_format or 'unknown'}, "
                    f"value: '{value}'"
                )
                format_invalid_fields.append(field_name)
                # Mark as None to trigger re-extraction
                extracted_data[field_name] = None
    
    # Check which fields failed extraction (None values) or have invalid formats
    missing_fields = [
        field_name for field_name, value in extracted_data.items()
        if value is None
    ]
    
    # If format validation failed, use LLM to re-extract and find alternative positions
    if format_invalid_fields:
        logger.info(
            f"Format validation failed for {len(format_invalid_fields)} fields: "
            f"{format_invalid_fields}. Re-extracting with LLM and finding alternative positions."
        )
        
        # Re-extract invalid fields using LLM
        llm_extracted = extract_with_llm(
            text_blocks,
            extraction_schema,
            fields_to_extract=format_invalid_fields,
        )
        # Benchmark: count per-field LLM usage (attempts)
        global _benchmark_llm_fields_used
        if _benchmark_enabled:
            _benchmark_llm_fields_used += len(format_invalid_fields)
        
        # Find positions of LLM-extracted values and store as alternative positions
        template = get_template(template_hash)
        if template:
            # Extract positions for LLM-extracted values
            llm_positions, _, _ = _extract_field_positions(text_blocks, llm_extracted)
            
            # Store alternative positions in template
            if "alternative_positions" not in template:
                template["alternative_positions"] = {}
            
            for field_name in format_invalid_fields:
                if field_name in llm_positions:
                    if field_name not in template["alternative_positions"]:
                        template["alternative_positions"][field_name] = []
                    
                    # Add as alternative position (avoid duplicates)
                    alt_pos = llm_positions[field_name]
                    is_duplicate = False
                    for existing_pos in template["alternative_positions"][field_name]:
                        if (existing_pos.get("x0") == alt_pos.get("x0") and
                            existing_pos.get("y0") == alt_pos.get("y0")):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        template["alternative_positions"][field_name].append(alt_pos)
                        logger.info(
                            f"Stored alternative position for '{field_name}' "
                            f"at ({alt_pos.get('x0')}, {alt_pos.get('y0')})"
                        )
            
            # Update extracted_data with LLM results
            # CRITICAL: Always incorporate LLM results, even if None
            # The LLM result should replace the invalid value
            for field_name in format_invalid_fields:
                if field_name in llm_extracted:
                    llm_value = llm_extracted[field_name]
                    # Always update with LLM result, even if None
                    extracted_data[field_name] = llm_value
                    if llm_value is not None:
                        logger.info(f"LLM re-extracted '{field_name}' after format validation failure: {llm_value}")
                    else:
                        logger.warning(f"LLM returned None for '{field_name}' after format validation failure")
                else:
                    # LLM didn't return this field - this shouldn't happen, but keep as None
                    logger.error(f"LLM did not return field '{field_name}' in re-extraction - this is unexpected")
                    # Keep as None (will be saved as null in JSON)
        
        # Remove fields already re-extracted via LLM from missing list
        missing_fields = [
            field_name for field_name in missing_fields
            if field_name not in format_invalid_fields
        ]
    
    # Before using LLM, ensure template knowledge was considered first
    if missing_fields:
        logger.info(
            f"Positional extraction failed for {len(missing_fields)} fields: {missing_fields}"
        )
        
        # Verify that template knowledge from templates.json was available and used
        template_knowledge_available = False
        if label:
            try:
                from template_manager import find_template_by_label_and_fields
                field_names = list(extraction_schema.keys())
                template_knowledge = find_template_by_label_and_fields(label, field_names)
                if template_knowledge:
                    template_knowledge_available = True
                    logger.info(
                        f"✓ Template knowledge from templates.json was available and used. "
                        f"Remaining missing fields: {missing_fields}"
                    )
                else:
                    logger.warning(
                        f"No template knowledge found in templates.json for label '{label}' "
                        f"with fields {field_names}"
                    )
            except Exception as e:
                logger.warning(f"Could not verify template knowledge: {str(e)}")
        
        # Use LLM only for remaining missing fields
        if missing_fields:
            logger.info(
                f"Template knowledge extraction still has {len(missing_fields)} missing fields: "
                f"{missing_fields}. Using incremental LLM extraction as fallback"
            )
            
            # Extract only missing fields using LLM
            llm_extracted = extract_with_llm(
                text_blocks,
                extraction_schema,
                fields_to_extract=missing_fields,
            )
            # Benchmark: count per-field LLM usage (attempts)
            if _benchmark_enabled:
                _benchmark_llm_fields_used += len(missing_fields)
            
            # Merge LLM results into extracted_data
            for field_name, value in llm_extracted.items():
                if value is not None:
                    extracted_data[field_name] = value
                    logger.debug(f"LLM extracted field '{field_name}': {value}")
        else:
            logger.info(
                f"Template knowledge successfully extracted all missing fields. "
                f"No LLM call needed."
            )
        
        # Update template with positions learned from LLM results
        template = get_template(template_hash)
        if template:
            # Re-extract positions for newly extracted fields
            updated_positions, updated_modes, updated_box_to_fields = _extract_field_positions(
                text_blocks, extracted_data
            )
            
            # Update template with new positions
            template["field_positions"].update(updated_positions)
            template["extraction_modes"].update(updated_modes)
            template["extracted_data"].update(extracted_data)
            
            # Learn multi-field box patterns if needed
            for box_text, field_list in updated_box_to_fields.items():
                if len(field_list) > 1 and box_text not in template.get("box_patterns", {}):
                    # New multi-field box discovered - learn pattern
                    from box_parser import segment_box_text, learn_box_pattern
                    segments = segment_box_text(box_text)
                    if len(segments) >= len(field_list):
                        pattern = learn_box_pattern(segments)
                        if "box_patterns" not in template:
                            template["box_patterns"] = {}
                        template["box_patterns"][box_text] = {
                            "pattern": pattern,
                            "segments": segments,
                            "fields": [field_name for field_name, _ in field_list],
                        }
                        logger.info(
                            f"Learned new pattern for box '{box_text[:50]}...': {pattern}"
                        )
    else:
        logger.info(
            f"Positional extraction successful for all {len(extraction_schema)} fields"
        )
    
    return extracted_data


def _get_output_filename(pdf_path: str) -> str:
    """Generate output filename based on PDF filename.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Output filename in results/ directory.
    """
    pdf_name = Path(pdf_path).stem  # Get filename without extension
    return f"results/{pdf_name}.json"


def _save_extraction_result(extracted_data: Dict[str, Any], output_file: str) -> None:
    """Save extraction result to JSON file.
    
    The output contains ONLY the fields from extraction_schema with their values.
    
    Args:
        extracted_data: Dictionary with extracted field values.
        output_file: Path to output JSON file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved extraction result to {output_file}")


def process_document(
    pdf_path: str,
    label: str,
    extraction_schema: Dict[str, str],
    output_file: str = "results/output.json",
) -> Dict[str, Any]:
    """Process a single document through the extraction pipeline.
    
    Args:
        pdf_path: Path to the PDF file.
        label: Document label/type identifier.
        extraction_schema: Dictionary mapping field names to extraction descriptions.
        output_file: Path to output JSON file for incremental results (legacy, not used for individual files).
        
    Returns:
        Dictionary containing extracted data (only extraction_schema fields).
    """
    logger.info(f"Processing document: {pdf_path} (label: {label})")
    # Declare benchmark globals before any assignment
    global _benchmark_first_time, _benchmark_times, _benchmark_llm_fields_used, _benchmark_total_fields
    import time
    _start_ts = time.perf_counter() if _benchmark_enabled else None
    # Benchmark: count total fields for this document once
    if _benchmark_enabled:
        _benchmark_total_fields += len(extraction_schema)
    
    # Generate output filename based on PDF name
    individual_output_file = _get_output_filename(pdf_path)
    
    # CRITICAL: Check cache FIRST - this is the first verification (if enabled)
    if _cache_enabled:
        cached_result = get_cached_result(pdf_path, extraction_schema, individual_output_file)
        if cached_result is not None:
            logger.info(f"Cache hit: using cached result for {Path(pdf_path).name}")
            return cached_result
    
    logger.info(f"Cache miss or disabled: processing document {Path(pdf_path).name}")
    
    result = {
        "pdf_path": pdf_path,
        "label": label,
        "extraction_method": None,
        "template_hash": None,
        "extracted_data": {},
    }
    
    try:
        doc_type = classify_document_type(label)
        logger.info(f"Document classified as: {doc_type}")
        
        if doc_type == "stricted":
            text_blocks = extract_text_with_coords(pdf_path)
            current_hash = compute_phash(pdf_path)
            template_hash = _find_matching_template(pdf_path, label=label, threshold=40)
            
            if template_hash and has_template(template_hash):
                logger.info(f"Using known template (hash: {template_hash})")
                result["extraction_method"] = "positional"
                result["template_hash"] = template_hash
                extracted_data = _extract_using_positions(
                    text_blocks, template_hash, extraction_schema, label=label
                )
            else:
                logger.info("Unknown template, using LLM for extraction and creating template")
                result["extraction_method"] = "llm_learning"
                result["template_hash"] = current_hash
                
                # Use Oracle to extract data with LLM
                extracted_data = extract_with_llm(text_blocks, extraction_schema)
                # Benchmark: LLM used for all fields (count attempts)
                if _benchmark_enabled:
                    _benchmark_llm_fields_used += len(extraction_schema)
                
                # Create and store template with ground truth and positions (gabarito)
                create_template(
                    template_hash=current_hash,
                    text_blocks=text_blocks,
                    extracted_data=extracted_data,
                    extraction_schema=extraction_schema,
                    pdf_path=pdf_path,
                    label=label,
                )
                logger.info(
                    f"Created template {current_hash} with extracted data and field positions"
                )
        
        else:
            logger.info("Flexible document, using simple text extraction with LLM")
            result["extraction_method"] = "llm_flexible"
            text_content = extract_text_simple(pdf_path)
            text_blocks = [
                {"text": text_content, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "page": 0}
            ]
            extracted_data = extract_with_llm(text_blocks, extraction_schema)
            # Benchmark: LLM used for all fields (count attempts)
            if _benchmark_enabled:
                _benchmark_llm_fields_used += len(extraction_schema)
        
        # Save individual JSON file with only extraction_schema fields
        _save_extraction_result(extracted_data, individual_output_file)
        
        # Save to cache for future use (if enabled)
        if _cache_enabled:
            save_to_cache(extracted_data, individual_output_file)
        
        # Keep legacy result structure for backward compatibility
        result["extracted_data"] = extracted_data
        _extraction_results.append(result)
        _save_results_incremental(output_file)
        
        # Record duration
        if _benchmark_enabled and _start_ts is not None:
            duration = time.perf_counter() - _start_ts
            _benchmark_times.append(duration)
            if _benchmark_first_time is None:
                _benchmark_first_time = duration
        return extracted_data
        
    except Exception as e:
        logger.error(f"Failed to process document {pdf_path}: {str(e)}", exc_info=True)
        # On error, save file with all fields as null
        extracted_data = {field: None for field in extraction_schema.keys()}
        _save_extraction_result(extracted_data, individual_output_file)
        
        # Don't save error results to cache - they should be reprocessed
        # save_to_cache(extracted_data, individual_output_file)
        
        result["error"] = str(e)
        result["extracted_data"] = extracted_data
        _extraction_results.append(result)
        _save_results_incremental(output_file)
        if _benchmark_enabled and _start_ts is not None:
            duration = time.perf_counter() - _start_ts
            _benchmark_times.append(duration)
            if _benchmark_first_time is None:
                _benchmark_first_time = duration
        return extracted_data


def _save_results_incremental(output_file: str) -> None:
    """Save extraction results incrementally to JSON file.
    
    Args:
        output_file: Path to output JSON file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not _incremental_enabled:
        return
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_extraction_results, f, ensure_ascii=False, indent=2)
    
    logger.debug(f"Saved {len(_extraction_results)} results to {output_file}")


def process_dataset(
    dataset: List[Dict[str, Any]], output_file: str = "results/output.json"
) -> List[Dict[str, Any]]:
    """Process a dataset of documents.
    
    Args:
        dataset: List of dataset entries, each containing label, extraction_schema, and pdf_path.
        output_file: Path to output JSON file.
        
    Returns:
        List of extraction results.
    """
    logger.info(f"Processing dataset with {len(dataset)} entries")
    
    for index, entry in enumerate(dataset, 1):
        pdf_path = entry.get("pdf_path")
        label = entry.get("label")
        extraction_schema = entry.get("extraction_schema", {})
        
        if not pdf_path or not label:
            logger.warning(f"Skipping invalid entry at index {index}")
            continue
        
        logger.info(f"Processing entry {index}/{len(dataset)}")
        process_document(pdf_path, label, extraction_schema, output_file)
    
    logger.info(f"Completed processing {len(dataset)} documents")
    return _extraction_results


def _process_entries_serial(entries: List[Dict[str, Any]], output_file: str) -> List[Dict[str, Any]]:
    """Process a list of dataset entries sequentially (helper for parallel per-label)."""
    results_local: List[Dict[str, Any]] = []
    for entry in entries:
        pdf_path = entry.get("pdf_path")
        label = entry.get("label")
        extraction_schema = entry.get("extraction_schema", {})
        if not pdf_path or not label:
            continue
        res = process_document(pdf_path, label, extraction_schema, output_file)
        results_local.append(res)
    return results_local


def process_dataset_parallel_by_label(
    dataset: List[Dict[str, Any]],
    output_file: str = "results/output.json",
    benchmark: bool = False,
    no_cache: bool = False,
) -> List[Dict[str, Any]]:
    """Process the dataset in parallel, one process per label queue.

    Groups entries by label (preserving first-seen label order), spawns one
    process per label, and processes each label's queue sequentially within its
    own process.
    """
    # Build label order (first occurrence) and mapping
    label_order: List[str] = []
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for entry in dataset:
        label = entry.get("label")
        if not label:
            continue
        if label not in by_label:
            by_label[label] = []
            label_order.append(label)
        by_label[label].append(entry)

    if not label_order:
        return []

    # Use a top-level worker function for multiprocessing picklability
    from pipeline import worker_process_label_queue as _worker

    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        import os
        max_workers = min(len(label_order), os.cpu_count() or 1)
        if max_workers < 1:
            max_workers = 1
    except Exception:
        max_workers = len(label_order)

    tasks = []
    results: List[Dict[str, Any]] = []
    bench_parts: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for label in label_order:
            entries = by_label[label]
            fut = executor.submit(_worker, entries, output_file, benchmark, no_cache)
            tasks.append(fut)

        for fut in as_completed(tasks):
            out = fut.result()
            if out and isinstance(out, dict):
                res_list = out.get("results") or []
                results.extend(res_list)
                stats = out.get("benchmark")
                if stats:
                    bench_parts.append(stats)
                times = out.get("times")
                if times:
                    # Temporarily stash in bench_parts with a marker
                    bench_parts.append({"__times__": list(times)})

    # Merge and write combined results file once (in parent)
    global _extraction_results
    _extraction_results.extend(results)
    _save_results_incremental(output_file)

    # Merge benchmark parts into current process benchmark state
    if benchmark and bench_parts:
        # Clear current and re-accumulate
        enable_benchmark(True)
        combined_times: List[float] = []
        for part in bench_parts:
            if "__times__" in part:
                combined_times.extend(part["__times__"])  # type: ignore
                continue
            # Reconstruct times approximating by using mean and count not possible; skip times
            # Aggregate only field counters; timings are already printed per overall
            global _benchmark_llm_fields_used, _benchmark_total_fields
            _benchmark_llm_fields_used += int(part.get("llm_fields_used") or 0)
            _benchmark_total_fields += int(part.get("total_fields") or 0)
        # Set combined times and first
        global _benchmark_times, _benchmark_first_time
        _benchmark_times.extend(combined_times)
        if combined_times and _benchmark_first_time is None:
            _benchmark_first_time = combined_times[0]

    return results


def worker_process_label_queue(
    entries: List[Dict[str, Any]],
    output_file: str,
    benchmark_flag: bool,
    no_cache_flag: bool,
) -> Dict[str, Any]:
    """Top-level worker to process a single label's queue in a separate process.

    Returns a dict with keys: "results" (list) and optional "benchmark" (dict).
    """
    # Silence logging in worker (no console noise during execution)
    try:
        import logging as _logging
        _logging.basicConfig(level=_logging.CRITICAL, force=True)
        _logging.disable(_logging.CRITICAL)
    except Exception:
        pass

    # Avoid cross-process contention on combined writes
    enable_incremental(False)
    if benchmark_flag:
        enable_benchmark(True)
    if no_cache_flag:
        enable_cache(False)
    results_local = _process_entries_serial(entries, output_file)
    stats = get_benchmark_stats() if benchmark_flag else None
    times = get_benchmark_times() if benchmark_flag else None
    return {"results": results_local, "benchmark": stats, "times": times}
