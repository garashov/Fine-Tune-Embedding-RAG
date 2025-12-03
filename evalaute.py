"""
Evaluation Script for Qwen3-Embedding Models
Evaluates original or fine-tuned models

Usage: Simply configure the variables in the CONFIG section and run the script
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import torch
from datasets import load_from_disk, load_dataset, Dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import InformationRetrievalEvaluator


# ============================================================================
# CONFIGURATION - Edit these variables to configure evaluation
# ============================================================================
# Path to your fine-tuned model (set to None if only evaluating baseline)
MODEL_PATH = "data/fine_tuning/20251203_114114/finetuned_qwen3_embedding"

# Baseline model to compare against
BASELINE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Evaluation options
EVALUATE_FINETUNED = True  # Evaluate the fine-tuned model
EVALUATE_BASELINE = True   # Evaluate the baseline model
COMPARE_MODELS = True      # Compare both models (requires both above to be True)

# Path to evaluation data directory
USE_LOCAL_EVAL_DATA = False
EVAL_DATA_DIR = None  # Auto-detects from MODEL_PATH/eval_data

USE_HF_DATASET = True
HF_DATASET_CONFIG = {
    "dataset_name": "philschmid/finanical-rag-embedding-dataset",
    "dataset_split": "train",
    "test_size": 0.1,
    "anchor_column": "question",
    "positive_column": "context",
    "max_samples": None,  # Limit samples if needed
    "seed": 42
}

# Output directory for results (None = same as MODEL_PATH)
OUTPUT_DIR = Path("data/eval_results")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging for evaluation
    
    Args:
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("qwen3_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Evaluator Creation
# ============================================================================

def create_evaluator_from_saved_data(
    eval_data_dir: Path,
    logger: logging.Logger
) -> InformationRetrievalEvaluator:
    """
    Create evaluator from saved evaluation data
    
    Args:
        eval_data_dir: Directory containing saved evaluation data
        logger: Logger instance
    
    Returns:
        Configured evaluator
    """
    logger.info(f"Loading evaluation data from: {eval_data_dir}")
    
    try:
        # Load datasets
        test_dataset = load_from_disk(str(eval_data_dir / "test_dataset"))
        corpus_dataset = load_from_disk(str(eval_data_dir / "corpus_dataset"))
        
        logger.info(f"Loaded {len(test_dataset)} test examples")
        logger.info(f"Loaded {len(corpus_dataset)} corpus documents")
        
        # Create corpus dictionary (id -> document)
        corpus = dict(
            zip(corpus_dataset["id"], corpus_dataset["positive"])
        )
        
        # Create queries dictionary (id -> question)
        queries = dict(
            zip(test_dataset["id"], test_dataset["anchor"])
        )
        
        # Create relevant documents mapping
        relevant_docs = {q_id: [q_id] for q_id in queries}
        
        # Create evaluator
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            score_functions={"cosine": cos_sim},
            name="ir-eval"
        )
        
        logger.info(f"Evaluator created with {len(queries)} queries and {len(corpus)} documents")
        
        return evaluator
        
    except Exception as e:
        logger.error(f"Failed to create evaluator: {str(e)}")
        raise


def create_evaluator_from_hf_dataset(
    config: Dict,
    logger: logging.Logger
) -> InformationRetrievalEvaluator:
    """
    Create evaluator from HuggingFace dataset
    
    Args:
        config: Dataset configuration dictionary
        logger: Logger instance
    
    Returns:
        Configured evaluator
    """
    logger.info(f"Loading dataset: {config['dataset_name']}")
    
    try:
        # Load dataset
        dataset = load_dataset(config['dataset_name'], split=config['dataset_split'])
        
        # Limit samples if specified
        if config['max_samples'] and len(dataset) > config['max_samples']:
            dataset = dataset.shuffle(seed=config['seed']).select(range(config['max_samples']))
            logger.info(f"Limited dataset to {config['max_samples']} samples")
        
        # Split into train/test
        dataset = dataset.train_test_split(
            test_size=config['test_size'],
            seed=config['seed']
        )
        test_dataset = dataset['test']
        
        logger.info(f"Using {len(test_dataset)} test examples")
        
        # Create corpus and queries
        corpus = {}
        queries = {}
        relevant_docs = {}
        
        for idx, example in enumerate(test_dataset):
            doc_id = f"doc_{idx}"
            query_id = f"query_{idx}"
            
            corpus[doc_id] = example[config['positive_column']]
            queries[query_id] = example[config['anchor_column']]
            relevant_docs[query_id] = [doc_id]
        
        # Create evaluator
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            score_functions={"cosine": cos_sim},
            name="ir-eval"
        )
        
        logger.info(f"Evaluator created with {len(queries)} queries and {len(corpus)} documents")
        
        return evaluator
        
    except Exception as e:
        logger.error(f"Failed to create evaluator from HF dataset: {str(e)}")
        raise


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model_path: str,
    evaluator: InformationRetrievalEvaluator,
    logger: logging.Logger,
    model_name: str = "model"
) -> Dict[str, float]:
    """
    Evaluate a single model
    
    Args:
        model_path: Path to model or model ID
        evaluator: Evaluator instance
        logger: Logger instance
        model_name: Name for logging purposes
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 80)
    logger.info(f"EVALUATING {model_name.upper()}")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory before loading: {gpu_mem_before:.2f} GB")
    
    model = None
    try:
        # Load model
        logger.info("Loading model...")
        model = SentenceTransformer(model_path, device=device)
        model.eval()
        logger.info("Model loaded successfully")
        
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after loading: {gpu_mem_after:.2f} GB")
            logger.info(f"GPU memory used by model: {gpu_mem_after - gpu_mem_before:.2f} GB")
        
        # Run evaluation
        logger.info("Running evaluation...")
        with torch.inference_mode():
            results = evaluator(model)
        
        logger.info("Evaluation complete!")
        
        # Log key metrics
        logger.info("\nKey Metrics:")
        logger.info("-" * 80)
        
        key_metrics = [
            "ir-eval_cosine_ndcg@10",
            "ir-eval_cosine_accuracy@10",
            "ir-eval_cosine_mrr@10",
            "ir-eval_cosine_precision@10",
            "ir-eval_cosine_recall@10"
        ]
        
        for metric in key_metrics:
            if metric in results:
                logger.info(f"  {metric:40s}: {results[metric]:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    
    finally:
        # Explicit cleanup to free memory
        if model is not None:
            logger.info("Cleaning up model from memory...")
            del model
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem_final = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after cleanup: {gpu_mem_final:.2f} GB")
            logger.info("GPU cache cleared")


def compare_models(
    baseline_results: Dict[str, float],
    finetuned_results: Dict[str, float],
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """
    Compare baseline and fine-tuned model results
    
    Args:
        baseline_results: Results from baseline model
        finetuned_results: Results from fine-tuned model
        logger: Logger instance
    
    Returns:
        Dictionary with comparison metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: BASELINE vs FINE-TUNED")
    logger.info("=" * 80)
    
    improvements = {}
    
    for key in baseline_results:
        if key in finetuned_results:
            baseline_val = baseline_results[key]
            finetuned_val = finetuned_results[key]
            
            # Calculate absolute and relative improvement
            abs_improvement = finetuned_val - baseline_val
            rel_improvement = (abs_improvement / baseline_val) * 100 if baseline_val != 0 else 0
            
            improvements[key] = {
                "baseline": baseline_val,
                "fine_tuned": finetuned_val,
                "absolute_improvement": abs_improvement,
                "relative_improvement_percent": rel_improvement
            }
    
    # Log key metrics
    logger.info("\nKey Metrics Comparison:")
    logger.info("-" * 80)
    
    key_metrics = [
        "ir-eval_cosine_ndcg@10",
        "ir-eval_cosine_accuracy@10",
        "ir-eval_cosine_mrr@10"
    ]
    
    for metric in key_metrics:
        if metric in improvements:
            info = improvements[metric]
            logger.info(f"\n{metric}:")
            logger.info(f"  Baseline:    {info['baseline']:.4f}")
            logger.info(f"  Fine-tuned:  {info['fine_tuned']:.4f}")
            logger.info(f"  Absolute Change:  {info['absolute_improvement']:+.4f}")
            logger.info(f"  Relative Change:  {info['relative_improvement_percent']:+.2f}%")
    
    return improvements


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main evaluation pipeline"""
    
    # Validate configuration
    if not EVALUATE_FINETUNED and not EVALUATE_BASELINE:
        raise ValueError("At least one of EVALUATE_FINETUNED or EVALUATE_BASELINE must be True")
    
    if COMPARE_MODELS and not (EVALUATE_FINETUNED and EVALUATE_BASELINE):
        raise ValueError("COMPARE_MODELS requires both EVALUATE_FINETUNED and EVALUATE_BASELINE to be True")
    
    if EVALUATE_FINETUNED and MODEL_PATH is None:
        raise ValueError("MODEL_PATH must be set when EVALUATE_FINETUNED is True")
    
    # Validate evaluation data source
    if USE_LOCAL_EVAL_DATA and USE_HF_DATASET:
        raise ValueError("Cannot use both local eval data and HuggingFace dataset. Set one to False.")
    
    if not USE_LOCAL_EVAL_DATA and not USE_HF_DATASET:
        raise ValueError("Must specify either USE_LOCAL_EVAL_DATA=True or USE_HF_DATASET=True")
    
    # Setup paths
    if MODEL_PATH:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
    
    # Determine output directory
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    elif MODEL_PATH:
        output_dir = Path(MODEL_PATH)
    else:
        output_dir = Path("eval_results")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / "logs" / f"evaluation_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("QWEN3-EMBEDDING EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  MODEL_PATH: {MODEL_PATH}")
    logger.info(f"  BASELINE_MODEL: {BASELINE_MODEL}")
    logger.info(f"  EVALUATE_FINETUNED: {EVALUATE_FINETUNED}")
    logger.info(f"  EVALUATE_BASELINE: {EVALUATE_BASELINE}")
    logger.info(f"  COMPARE_MODELS: {COMPARE_MODELS}")
    logger.info(f"  USE_LOCAL_EVAL_DATA: {USE_LOCAL_EVAL_DATA}")
    logger.info(f"  USE_HF_DATASET: {USE_HF_DATASET}")
    logger.info(f"  Output directory: {output_dir}")
    
    try:
        # Create evaluator based on data source
        if USE_LOCAL_EVAL_DATA:
            # Determine eval_data path
            if EVAL_DATA_DIR:
                eval_data_dir = Path(EVAL_DATA_DIR)
            elif MODEL_PATH:
                eval_data_dir = Path(MODEL_PATH) / "eval_data"
            else:
                raise ValueError("EVAL_DATA_DIR must be specified when MODEL_PATH is None")
            
            if not eval_data_dir.exists():
                raise ValueError(f"Evaluation data directory does not exist: {eval_data_dir}")
            
            logger.info(f"\nUsing local evaluation data from: {eval_data_dir}")
            evaluator = create_evaluator_from_saved_data(eval_data_dir, logger)
        
        else:  # USE_HF_DATASET
            logger.info(f"\nUsing HuggingFace dataset: {HF_DATASET_CONFIG['dataset_name']}")
            evaluator = create_evaluator_from_hf_dataset(HF_DATASET_CONFIG, logger)
        
        results = {}
        
        # NOTE: Models are evaluated sequentially to avoid memory issues
        # Each model is fully cleaned up before loading the next one
        
        # Evaluate baseline if requested
        if EVALUATE_BASELINE:
            baseline_results = evaluate_model(
                BASELINE_MODEL,
                evaluator,
                logger,
                model_name="baseline"
            )
            results["baseline"] = baseline_results
            
            # Save baseline results
            baseline_results_path = output_dir / f"baseline_results_{timestamp}.json"
            baseline_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_results_path, 'w') as f:
                json.dump(baseline_results, f, indent=2)
            logger.info(f"\nBaseline results saved to: {baseline_results_path}")
        
        # Evaluate fine-tuned model if requested
        if EVALUATE_FINETUNED and MODEL_PATH:
            finetuned_results = evaluate_model(
                str(model_path),
                evaluator,
                logger,
                model_name="fine-tuned"
            )
            results["fine_tuned"] = finetuned_results
            
            # Save fine-tuned results
            finetuned_results_path = output_dir / f"finetuned_results_{timestamp}.json"
            finetuned_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(finetuned_results_path, 'w') as f:
                json.dump(finetuned_results, f, indent=2)
            logger.info(f"\nFine-tuned results saved to: {finetuned_results_path}")
        
        # Compare models if both were evaluated
        if COMPARE_MODELS and "baseline" in results and "fine_tuned" in results:
            improvements = compare_models(
                results["baseline"],
                results["fine_tuned"],
                logger
            )
            
            # Save comparison results
            comparison_path = output_dir / f"comparison_{timestamp}.json"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            with open(comparison_path, 'w') as f:
                json.dump(improvements, f, indent=2)
            logger.info(f"\nComparison results saved to: {comparison_path}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Logs saved to: {log_file}")
        
        # Print quick summary of results
        if COMPARE_MODELS and "baseline" in results and "fine_tuned" in results:
            logger.info("\nQuick Summary:")
            logger.info("-" * 80)
            key_metric = "ir-eval_cosine_ndcg@10"
            if key_metric in improvements:
                info = improvements[key_metric]
                logger.info(f"NDCG@10: {info['baseline']:.4f} -> {info['fine_tuned']:.4f} "
                           f"({info['relative_improvement_percent']:+.2f}%)")
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("EVALUATION FAILED")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()