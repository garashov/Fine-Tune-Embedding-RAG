"""
Production-Grade Evaluation Script for Qwen3-Embedding Models

This module provides a robust evaluation framework for embedding models with:
- Configuration-driven architecture
- Class-based design for reusability
- Comprehensive error handling
- Memory-efficient model loading/unloading
- Modular components ready for extraction to utils
"""
from pathlib import Path
from typing import Dict
from datetime import datetime
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.utils.logging import EvaluationLogger
from src.utils.model_evaluator import ModelEvaluator
from src.utils.model_comparator import ModelComparator
from src.utils.evaluator_factory import EvaluatorFactory
from src.utils.data_models import EvaluationResults, ComparisonResults
from src.config.config import (
    EVAL_MODEL_FT_PATH,
    EVAL_MODEL_BASELINE_ID,
    EVAL_EVALUATE_FINETUNED,
    EVAL_EVALUATE_BASELINE,
    EVAL_COMPARE_MODELS,
    EVAL_DEVICE,
    EVAL_DATA_USE_LOCAL_EVAL_DATA,
    EVAL_DATA_USE_HUGGINGFACE_DATASET,
    EVAL_DATA_LOCAL_EVAL_DATA_DIR,
    EVAL_DATA_HF_DATASET_NAME,
    EVAL_OUTPUT_DIR,
    EVAL_OUTPUT_LOGGING_SAVE_TO_FILE,
    EVAL_OUTPUT_LOGGING_LOG_SUBDIR,
    EVAL_OUTPUT_RESULTS_INCLUDE_TIMESTAMP,
    EVAL_OUTPUT_RESULTS_BASELINE_RESULTS,
    EVAL_OUTPUT_RESULTS_FINETUNED_RESULTS,
    EVAL_OUTPUT_RESULTS_COMPARISON_RESULTS,
)


class EvaluationPipeline:
    """
    Main pipeline orchestrating the entire evaluation process
    """
    
    def __init__(self):
        """Initialize evaluation pipeline with configuration"""
        self._validate_configuration()
        self._setup_paths()
        self._setup_logging()
        
        # Initialize components
        self.evaluator_factory = EvaluatorFactory()
        self.model_evaluator = ModelEvaluator(logger=self.logger)
        self.model_comparator = ModelComparator(logger=self.logger)
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings"""
        if not EVAL_EVALUATE_FINETUNED and not EVAL_EVALUATE_BASELINE:
            raise ValueError(
                "At least one of EVAL_EVALUATE_FINETUNED or EVAL_EVALUATE_BASELINE must be True"
            )
        
        if EVAL_COMPARE_MODELS and not (EVAL_EVALUATE_FINETUNED and EVAL_EVALUATE_BASELINE):
            raise ValueError(
                "EVAL_COMPARE_MODELS requires both models to be evaluated"
            )
        
        if EVAL_EVALUATE_FINETUNED and not EVAL_MODEL_FT_PATH:
            raise ValueError(
                "EVAL_MODEL_FT_PATH must be set when EVAL_EVALUATE_FINETUNED is True"
            )
        
        if EVAL_DATA_USE_LOCAL_EVAL_DATA and EVAL_DATA_USE_HUGGINGFACE_DATASET:
            raise ValueError(
                "Cannot use both local eval data and HuggingFace dataset"
            )
        
        if not EVAL_DATA_USE_LOCAL_EVAL_DATA and not EVAL_DATA_USE_HUGGINGFACE_DATASET:
            raise ValueError(
                "Must specify either local eval data or HuggingFace dataset"
            )
    
    def _setup_paths(self) -> None:
        """Setup file paths for evaluation"""
        # Model path
        if EVAL_MODEL_FT_PATH:
            self.model_path = Path(EVAL_MODEL_FT_PATH)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
        else:
            self.model_path = None
        
        # Output directory
        if EVAL_OUTPUT_DIR:
            self.output_dir = Path(EVAL_OUTPUT_DIR)
        elif self.model_path:
            self.output_dir = self.model_path
        else:
            self.output_dir = Path("eval_results")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = None
        if EVAL_OUTPUT_LOGGING_SAVE_TO_FILE:
            log_dir = self.output_dir / EVAL_OUTPUT_LOGGING_LOG_SUBDIR
            log_file = log_dir / f"evaluation_{self.timestamp}.log"
        
        logger_wrapper = EvaluationLogger(log_file=log_file)
        self.logger = logger_wrapper.get_logger()
        
        # Log configuration
        self._log_configuration()
    
    def _log_configuration(self) -> None:
        """Log current configuration"""
        self.logger.info("=" * 80)
        self.logger.info("QWEN3-EMBEDDING EVALUATION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info("Configuration:")
        self.logger.info(f"  Model (Fine-tuned):   {EVAL_MODEL_FT_PATH}")
        self.logger.info(f"  Model (Baseline):     {EVAL_MODEL_BASELINE_ID}")
        self.logger.info(f"  Evaluate Fine-tuned:  {EVAL_EVALUATE_FINETUNED}")
        self.logger.info(f"  Evaluate Baseline:    {EVAL_EVALUATE_BASELINE}")
        self.logger.info(f"  Compare Models:       {EVAL_COMPARE_MODELS}")
        self.logger.info(f"  Use Local Data:       {EVAL_DATA_USE_LOCAL_EVAL_DATA}")
        self.logger.info(f"  Use HF Dataset:       {EVAL_DATA_USE_HUGGINGFACE_DATASET}")
        self.logger.info(f"  Output Directory:     {self.output_dir}")
        self.logger.info(f"  Device:               {EVAL_DEVICE or 'auto'}")
    
    def _create_evaluator(self) -> InformationRetrievalEvaluator:
        """Create evaluator based on data source configuration"""
        if EVAL_DATA_USE_LOCAL_EVAL_DATA:
            # Determine eval data directory
            if EVAL_DATA_LOCAL_EVAL_DATA_DIR:
                eval_data_dir = Path(EVAL_DATA_LOCAL_EVAL_DATA_DIR)
            elif self.model_path:
                eval_data_dir = self.model_path / "eval_data"
            else:
                raise ValueError(
                    "EVAL_DATA_LOCAL_EVAL_DATA_DIR must be specified when model path is None"
                )
            
            self.logger.info(f"\nUsing local evaluation data: {eval_data_dir}")
            return self.evaluator_factory.from_local_data(
                eval_data_dir=eval_data_dir,
                logger=self.logger
            )
        
        else:  # Use HuggingFace dataset
            self.logger.info(f"\nUsing HuggingFace dataset: {EVAL_DATA_HF_DATASET_NAME}")
            return self.evaluator_factory.from_huggingface_dataset(logger=self.logger)
    
    def _get_results_filename(self, base_name: str) -> Path:
        """Generate results filename with optional timestamp"""
        if EVAL_OUTPUT_RESULTS_INCLUDE_TIMESTAMP:
            filename = f"{base_name}_{self.timestamp}.json"
        else:
            filename = f"{base_name}.json"
        return self.output_dir / filename
    
    def run(self) -> Dict[str, EvaluationResults]:
        """
        Execute the complete evaluation pipeline
        
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Create evaluator
            evaluator = self._create_evaluator()
            
            results = {}
            
            # Evaluate baseline model
            if EVAL_EVALUATE_BASELINE:
                baseline_results = self.model_evaluator.evaluate(
                    model_path=EVAL_MODEL_BASELINE_ID,
                    evaluator=evaluator,
                    model_name="baseline"
                )
                results["baseline"] = baseline_results
                
                # Save baseline results
                baseline_path = self._get_results_filename(
                    EVAL_OUTPUT_RESULTS_BASELINE_RESULTS
                )
                baseline_results.save(baseline_path)
                self.logger.info(f"\nBaseline results saved: {baseline_path}")
            
            # Evaluate fine-tuned model
            if EVAL_EVALUATE_FINETUNED and self.model_path:
                finetuned_results = self.model_evaluator.evaluate(
                    model_path=str(self.model_path),
                    evaluator=evaluator,
                    model_name="fine-tuned"
                )
                results["fine_tuned"] = finetuned_results
                
                # Save fine-tuned results
                finetuned_path = self._get_results_filename(
                    EVAL_OUTPUT_RESULTS_FINETUNED_RESULTS
                )
                finetuned_results.save(finetuned_path)
                self.logger.info(f"\nFine-tuned results saved: {finetuned_path}")
            
            # Compare models
            if EVAL_COMPARE_MODELS and "baseline" in results and "fine_tuned" in results:
                comparison = self.model_comparator.compare(
                    baseline_results=results["baseline"],
                    finetuned_results=results["fine_tuned"]
                )
                
                # Save comparison
                comparison_path = self._get_results_filename(
                    EVAL_OUTPUT_RESULTS_COMPARISON_RESULTS
                )
                comparison.save(comparison_path)
                self.logger.info(f"\nComparison results saved: {comparison_path}")
                
                # Log summary
                self._log_summary(comparison)
            
            self._log_completion()
            
            return results
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("EVALUATION FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            raise
    
    def _log_summary(self, comparison: ComparisonResults) -> None:
        """Log quick summary of comparison results"""
        self.logger.info("=" * 80)
        self.logger.info("QUICK SUMMARY")
        self.logger.info("=" * 80)
        
        key_metric = "ir-eval_cosine_ndcg@10"
        if key_metric in comparison.improvements:
            info = comparison.improvements[key_metric]
            self.logger.info(
                f"NDCG@10: {info['baseline']:.4f} -> {info['fine_tuned']:.4f} "
                f"({info['relative_improvement_percent']:+.2f}%)"
            )
    
    def _log_completion(self) -> None:
        """Log completion message"""
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Results directory: {self.output_dir}")
        self.logger.info(f"Timestamp: {self.timestamp}")