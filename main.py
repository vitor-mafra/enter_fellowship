"""Main entry point for the document extraction system."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from pipeline import process_dataset

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the dataset JSON file.
    
    Args:
        dataset_path: Path to the dataset.json file.
        
    Returns:
        List of dictionaries containing document metadata and extraction schemas.
        
    Raises:
        FileNotFoundError: If the dataset file does not exist.
        json.JSONDecodeError: If the dataset file is not valid JSON.
        ValueError: If the dataset structure is invalid.
    """
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array")
        
        logger.info(f"Loaded dataset from {dataset_path}: {len(data)} entries")
        return data
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in dataset file {dataset_path}: {str(e)}", e.doc, e.pos
        ) from e


def main() -> None:
    """Main entry point that orchestrates the document extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Process PDF documents from a dataset JSON file"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset JSON file (e.g., dataset.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output to console",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure per-file execution time and LLM usage; print summary at end",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run processing in parallel: one process per label queue",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache read/write for this run",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.CRITICAL,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        logging.disable(logging.CRITICAL)
    
    try:
        logger.info("Starting document extraction pipeline")
        logger.info(f"Loading dataset from: {args.dataset_path}")
        
        dataset = load_dataset(args.dataset_path)
        logger.info(f"Processing {len(dataset)} document entries")
        
        # Enable benchmark in parent process if requested
        if args.benchmark:
            from pipeline import enable_benchmark
            enable_benchmark(True)
        if args.no_cache:
            from pipeline import enable_cache
            enable_cache(False)
        
        if args.parallel:
            from pipeline import process_dataset_parallel_by_label
            results = process_dataset_parallel_by_label(
                dataset,
                output_file="results/output.json",
                benchmark=args.benchmark,
                no_cache=args.no_cache,
            )
        else:
            from pipeline import process_dataset
            results = process_dataset(dataset, output_file="results/output.json")
        
        # Print benchmark summary at end (only when requested)
        if args.benchmark:
            from pipeline import get_benchmark_stats
            stats = get_benchmark_stats()
            # Always print, even if verbose is off (requested by user)
            print("\n=== Benchmark Summary ===")
            print(f"Files: {stats['count']}")
            print(f"Total: {stats['total_s']:.3f}s | First: {stats['first_s']:.3f}s")
            print(f"Min: {stats['min_s']:.3f}s | Median: {stats['median_s']:.3f}s | Mean: {stats['mean_s']:.3f}s | Std: {stats['std_s']:.3f}s | Max: {stats['max_s']:.3f}s")
            print("\n=== LLM Usage ===")
            print(f"Fields via LLM: {stats['llm_fields_used']} / {stats['total_fields']}")
        
        logger.info("Dataset processing completed")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
