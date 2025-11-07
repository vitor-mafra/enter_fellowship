"""Evaluator to compare extraction results with ground truth (gabarito)."""

import json
import logging
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
VERBOSE_REPORT = True


def normalize_text(text: Any) -> str:
    """Normalize text for comparison: lowercase, remove accents, remove newlines.
    
    Args:
        text: Text to normalize (can be None, str, int, etc.).
        
    Returns:
        Normalized string.
    """
    if text is None:
        return ""
    
    # Convert to string
    text_str = str(text).strip()
    
    # Remove newlines and extra whitespace
    text_str = " ".join(text_str.split())
    
    # Convert to lowercase
    text_str = text_str.lower()
    
    # Remove accents (normalize to NFD and remove combining characters)
    text_str = unicodedata.normalize("NFD", text_str)
    text_str = "".join(c for c in text_str if unicodedata.category(c) != "Mn")
    
    return text_str


def compare_field(predicted: Any, ground_truth: Any) -> bool:
    """Compare two field values after normalization.
    
    Args:
        predicted: Predicted value.
        ground_truth: Ground truth value.
        
    Returns:
        True if values match after normalization, False otherwise.
    """
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    return pred_norm == gt_norm


def compare_extractions(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Tuple[Dict[str, bool], Dict[str, Tuple[Any, Any]]]:
    """Compare predicted extraction with ground truth.
    
    Args:
        predicted: Dictionary with predicted field values.
        ground_truth: Dictionary with ground truth field values.
        
    Returns:
        Tuple of:
        - Dictionary mapping field names to match status (True/False).
        - Dictionary mapping field names to (predicted, ground_truth) tuples for errors.
    """
    matches = {}
    errors = {}
    
    # Get all unique field names from both dictionaries
    all_fields = set(predicted.keys()) | set(ground_truth.keys())
    
    for field in all_fields:
        pred_value = predicted.get(field)
        gt_value = ground_truth.get(field)
        
        is_match = compare_field(pred_value, gt_value)
        matches[field] = is_match
        
        if not is_match:
            errors[field] = (pred_value, gt_value)
    
    return matches, errors


def evaluate_results(
    results_dir: str = "results",
    oracle_dir: str = "oracle_results",
) -> Dict[str, Any]:
    """Evaluate extraction results against ground truth.
    
    Args:
        results_dir: Directory containing predicted results.
        oracle_dir: Directory containing ground truth (gabarito).
        
    Returns:
        Dictionary with evaluation metrics and errors.
    """
    results_path = Path(results_dir)
    oracle_path = Path(oracle_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}
    
    if not oracle_path.exists():
        logger.error(f"Oracle directory not found: {oracle_dir}")
        return {}
    
    # Find all JSON files in oracle_results (ground truth)
    oracle_files = {f.stem: f for f in oracle_path.glob("*.json")}
    
    # Find matching files in results
    results_files = {f.stem: f for f in results_path.glob("*.json")}
    
    # Statistics
    total_files = 0
    total_fields = 0
    correct_fields = 0
    file_matches = {}  # filename -> (total_fields, correct_fields)
    field_errors = defaultdict(list)  # field_name -> [(filename, predicted, ground_truth), ...]
    missing_files = []
    extra_files = []
    
    # Compare files that exist in both directories
    common_files = set(oracle_files.keys()) & set(results_files.keys())
    
    for filename in sorted(common_files):
        total_files += 1
        
        try:
            # Load predicted result
            with open(results_files[filename], "r", encoding="utf-8") as f:
                predicted = json.load(f)
            
            # Load ground truth
            with open(oracle_files[filename], "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            
            # Compare
            matches, errors = compare_extractions(predicted, ground_truth)
            
            # Update statistics
            file_total = len(matches)
            file_correct = sum(1 for m in matches.values() if m)
            
            total_fields += file_total
            correct_fields += file_correct
            
            file_matches[filename] = (file_total, file_correct)
            
            # Collect errors
            for field, (pred_val, gt_val) in errors.items():
                field_errors[field].append((filename, pred_val, gt_val))
        
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")
            continue
    
    # Find missing and extra files
    missing_files = sorted(set(oracle_files.keys()) - set(results_files.keys()))
    extra_files = sorted(set(results_files.keys()) - set(oracle_files.keys()))
    
    # Calculate accuracy
    accuracy = (correct_fields / total_fields * 100) if total_fields > 0 else 0.0
    
    return {
        "total_files": total_files,
        "total_fields": total_fields,
        "correct_fields": correct_fields,
        "accuracy": accuracy,
        "file_matches": file_matches,
        "field_errors": dict(field_errors),
        "missing_files": missing_files,
        "extra_files": extra_files,
    }


def print_evaluation_report(evaluation: Dict[str, Any]) -> None:
    """Print a detailed evaluation report.
    
    Args:
        evaluation: Evaluation results dictionary.
    """
    if not VERBOSE_REPORT:
        return
    
    print("\n" + "=" * 80)
    print("AVALIAÇÃO DE ACURÁCIA")
    print("=" * 80)
    
    total_files = evaluation.get("total_files", 0)
    total_fields = evaluation.get("total_fields", 0)
    correct_fields = evaluation.get("correct_fields", 0)
    accuracy = evaluation.get("accuracy", 0.0)
    
    print(f"\n📊 RESUMO GERAL:")
    print(f"  Arquivos processados: {total_files}")
    print(f"  Total de campos: {total_fields}")
    print(f"  Campos corretos: {correct_fields}")
    print(f"  Campos incorretos: {total_fields - correct_fields}")
    print(f"  Acurácia: {accuracy:.2f}%")
    
    # Per-file accuracy
    file_matches = evaluation.get("file_matches", {})
    if file_matches:
        print(f"\n📁 ACURÁCIA POR ARQUIVO:")
        for filename, (total, correct) in sorted(file_matches.items()):
            file_acc = (correct / total * 100) if total > 0 else 0.0
            status = "✓" if correct == total else "✗"
            print(f"  {status} {filename}: {correct}/{total} ({file_acc:.1f}%)")
    
    # Missing files
    missing_files = evaluation.get("missing_files", [])
    if missing_files:
        print(f"\n⚠️  ARQUIVOS FALTANDO EM RESULTS ({len(missing_files)}):")
        for filename in missing_files[:10]:  # Show first 10
            print(f"  - {filename}.json")
        if len(missing_files) > 10:
            print(f"  ... e mais {len(missing_files) - 10} arquivos")
    
    # Extra files
    extra_files = evaluation.get("extra_files", [])
    if extra_files:
        print(f"\nℹ️  ARQUIVOS EXTRAS EM RESULTS ({len(extra_files)}):")
        for filename in extra_files[:10]:  # Show first 10
            print(f"  - {filename}.json")
        if len(extra_files) > 10:
            print(f"  ... e mais {len(extra_files) - 10} arquivos")
    
    # Field errors
    field_errors = evaluation.get("field_errors", {})
    if field_errors:
        print(f"\n❌ ERROS POR CAMPO:")
        for field, errors in sorted(field_errors.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n  Campo: {field} ({len(errors)} erros)")
            for filename, pred_val, gt_val in errors[:5]:  # Show first 5 errors per field
                pred_str = str(pred_val)[:50] if pred_val is not None else "null"
                gt_str = str(gt_val)[:50] if gt_val is not None else "null"
                print(f"    {filename}:")
                print(f"      Previsto:  {pred_str}")
                print(f"      Esperado:  {gt_str}")
            if len(errors) > 5:
                print(f"    ... e mais {len(errors) - 5} erros")
    
    print("\n" + "=" * 80)


def main():
    """Main function to run evaluation."""
    logger.info("Starting evaluation...")
    
    evaluation = evaluate_results()
    
    if evaluation:
        print_evaluation_report(evaluation)
    else:
        logger.error("Evaluation failed or no files found")


if __name__ == "__main__":
    main()

