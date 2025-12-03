import torch
import logging
from typing import Dict, Optional, List
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.utils.gpu_utils import GPUMemoryManager
from src.utils.data_models import EvaluationResults
from src.config.config import (
    EVAL_DEVICE,
    EVAL_METRICS_KEY_METRICS,
    EVAL_PERF_CLEAR_CUDA_CACHE,
    EVAL_PERF_FORCE_GARBAGE_COLLECTION,
    EVAL_PERF_REPORT_GPU_MEMORY,
    EVAL_ADVANCED_USE_INFERENCE_MODE,
    EVAL_ADVANCED_EVAL_MODE,
)


class ModelEvaluator:
    """
    Handles model loading, evaluation, and cleanup
    Can be moved to core/model_evaluator.py
    """
    
    def __init__(
        self,
        device: Optional[str] = EVAL_DEVICE,
        use_inference_mode: bool = EVAL_ADVANCED_USE_INFERENCE_MODE,
        eval_mode: bool = EVAL_ADVANCED_EVAL_MODE,
        report_gpu_memory: bool = EVAL_PERF_REPORT_GPU_MEMORY,
        clear_cuda_cache: bool = EVAL_PERF_CLEAR_CUDA_CACHE,
        force_gc: bool = EVAL_PERF_FORCE_GARBAGE_COLLECTION,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize model evaluator
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_inference_mode: Whether to use torch.inference_mode()
            eval_mode: Whether to set model to eval mode
            report_gpu_memory: Whether to report GPU memory usage
            clear_cuda_cache: Whether to clear CUDA cache after evaluation
            force_gc: Whether to force garbage collection
            logger: Optional logger instance
        """
        self.device = GPUMemoryManager.get_device(device)
        self.use_inference_mode = use_inference_mode
        self.eval_mode = eval_mode
        self.report_gpu_memory = report_gpu_memory
        self.clear_cuda_cache = clear_cuda_cache
        self.force_gc = force_gc
        self.logger = logger or logging.getLogger(__name__)
    
    def _log_gpu_memory(self, stage: str) -> float:
        """Log GPU memory at a specific stage"""
        if self.report_gpu_memory and GPUMemoryManager.is_available():
            mem_gb = GPUMemoryManager.get_memory_allocated_gb()
            self.logger.info(f"GPU memory {stage}: {mem_gb:.2f} GB")
            return mem_gb
        return 0.0
    
    def evaluate(
        self,
        model_path: str,
        evaluator: InformationRetrievalEvaluator,
        model_name: str = "model",
        key_metrics: List[str] = EVAL_METRICS_KEY_METRICS
    ) -> EvaluationResults:
        """
        Evaluate a single model
        
        Args:
            model_path: Path to model or model identifier
            evaluator: Configured evaluator instance
            model_name: Name for logging and results
            key_metrics: List of key metrics to highlight
        
        Returns:
            EvaluationResults object containing all metrics and metadata
        """
        self.logger.info("=" * 80)
        self.logger.info(f"EVALUATING {model_name.upper()}")
        self.logger.info("=" * 80)
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Device: {self.device}")
        
        start_time = datetime.now()
        mem_before = self._log_gpu_memory("before loading")
        
        model = None
        try:
            # Load model
            self.logger.info("Loading model...")
            model = SentenceTransformer(model_path, device=self.device)
            
            if self.eval_mode:
                model.eval()
            
            self.logger.info("Model loaded successfully")
            
            mem_after = self._log_gpu_memory("after loading")
            if self.report_gpu_memory and GPUMemoryManager.is_available():
                self.logger.info(f"Model memory usage: {mem_after - mem_before:.2f} GB")
            
            # Run evaluation
            self.logger.info("Running evaluation...")
            
            if self.use_inference_mode:
                with torch.inference_mode():
                    metrics = evaluator(model)
            else:
                metrics = evaluator(model)
            
            end_time = datetime.now()
            eval_time = (end_time - start_time).total_seconds()
            
            self.logger.info("Evaluation complete!")
            self.logger.info(f"Evaluation time: {eval_time:.2f} seconds")
            
            # Log key metrics
            self._log_key_metrics(metrics, key_metrics)
            
            # Create results object
            results = EvaluationResults(
                model_name=model_name,
                model_path=model_path,
                metrics=metrics,
                timestamp=start_time.strftime("%Y%m%d_%H%M%S"),
                device=self.device,
                evaluation_time_seconds=eval_time
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            self._cleanup_model(model)
    
    def _log_key_metrics(self, metrics: Dict[str, float], key_metrics: List[str]) -> None:
        """Log key metrics from evaluation results"""
        self.logger.info("\nKey Metrics:")
        self.logger.info("-" * 80)
        
        for metric in key_metrics:
            if metric in metrics:
                self.logger.info(f"  {metric:40s}: {metrics[metric]:.4f}")
    
    def _cleanup_model(self, model: Optional[SentenceTransformer]) -> None:
        """Clean up model and free memory"""
        if model is not None:
            self.logger.info("Cleaning up model from memory...")
            del model
        
        if self.force_gc:
            GPUMemoryManager.force_garbage_collection()
        
        if self.clear_cuda_cache and GPUMemoryManager.is_available():
            GPUMemoryManager.clear_cache()
            self._log_gpu_memory("after cleanup")

