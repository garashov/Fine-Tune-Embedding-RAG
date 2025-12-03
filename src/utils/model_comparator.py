import logging
from typing import Dict, Optional, List
from datetime import datetime

from src.utils.data_models import EvaluationResults, ComparisonResults
from src.config.config import EVAL_METRICS_KEY_METRICS


class ModelComparator:
    """
    Handles comparison between baseline and fine-tuned models
    Can be moved to core/model_comparator.py
    """
    
    def __init__(
        self,
        key_metrics: List[str] = EVAL_METRICS_KEY_METRICS,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize model comparator
        
        Args:
            key_metrics: List of key metrics to highlight in comparison
            logger: Optional logger instance
        """
        self.key_metrics = key_metrics
        self.logger = logger or logging.getLogger(__name__)
    
    def compare(
        self,
        baseline_results: EvaluationResults,
        finetuned_results: EvaluationResults
    ) -> ComparisonResults:
        """
        Compare baseline and fine-tuned model results
        
        Args:
            baseline_results: Results from baseline model
            finetuned_results: Results from fine-tuned model
        
        Returns:
            ComparisonResults object with detailed comparison
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPARISON: BASELINE vs FINE-TUNED")
        self.logger.info("=" * 80)
        
        improvements = self._calculate_improvements(
            baseline_results.metrics,
            finetuned_results.metrics
        )
        
        self._log_key_comparisons(improvements)
        
        comparison_results = ComparisonResults(
            baseline_results=baseline_results,
            finetuned_results=finetuned_results,
            improvements=improvements,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        return comparison_results
    
    def _calculate_improvements(
        self,
        baseline_metrics: Dict[str, float],
        finetuned_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate improvements for all metrics"""
        improvements = {}
        
        for key in baseline_metrics:
            if key in finetuned_metrics:
                baseline_val = baseline_metrics[key]
                finetuned_val = finetuned_metrics[key]
                
                abs_improvement = finetuned_val - baseline_val
                rel_improvement = (abs_improvement / baseline_val * 100) if baseline_val != 0 else 0
                
                improvements[key] = {
                    "baseline": baseline_val,
                    "fine_tuned": finetuned_val,
                    "absolute_improvement": abs_improvement,
                    "relative_improvement_percent": rel_improvement
                }
        
        return improvements
    
    def _log_key_comparisons(self, improvements: Dict[str, Dict[str, float]]) -> None:
        """Log key metric comparisons"""
        self.logger.info("\nKey Metrics Comparison:")
        self.logger.info("-" * 80)
        
        for metric in self.key_metrics:
            if metric in improvements:
                info = improvements[metric]
                self.logger.info(f"\n{metric}:")
                self.logger.info(f"  Baseline:         {info['baseline']:.4f}")
                self.logger.info(f"  Fine-tuned:       {info['fine_tuned']:.4f}")
                self.logger.info(f"  Absolute Change:  {info['absolute_improvement']:+.4f}")
                self.logger.info(f"  Relative Change:  {info['relative_improvement_percent']:+.2f}%")

