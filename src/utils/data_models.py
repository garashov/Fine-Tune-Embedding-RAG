import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict


# ============================================================================
# Evalaution Data Models
# ============================================================================
@dataclass
class TrainingMetrics:
    """Container for training metrics and results"""
    train_runtime: float
    train_samples_per_second: float
    train_steps_per_second: float
    total_flos: float
    train_loss: float
    epoch: float
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def save(self, filepath: Path) -> None:
        """Save metrics to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

@dataclass
class ModelInfo:
    """Container for model information"""
    base_model: str
    total_parameters: int
    trainable_parameters: int
    trainable_percentage: float
    lora_rank: int
    lora_alpha: int
    target_modules: List[str]
    timestamp: str
    device: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def save(self, filepath: Path) -> None:
        """Save model info to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Evaluation Data Models
# ============================================================================
@dataclass
class EvaluationResults:
    """Container for evaluation results with metadata"""
    model_name: str
    model_path: str
    metrics: Dict[str, float]
    timestamp: str
    device: str
    evaluation_time_seconds: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "device": self.device,
            "evaluation_time_seconds": self.evaluation_time_seconds
        }
    
    def save(self, filepath: Path) -> None:
        """Save results to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ComparisonResults:
    """Container for model comparison results"""
    baseline_results: EvaluationResults
    finetuned_results: EvaluationResults
    improvements: Dict[str, Dict[str, float]]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "baseline": self.baseline_results.to_dict(),
            "finetuned": self.finetuned_results.to_dict(),
            "improvements": self.improvements,
            "timestamp": self.timestamp
        }
    
    def save(self, filepath: Path) -> None:
        """Save comparison results to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# HF Push Data Models
# ============================================================================
@dataclass
class PushResult:
    """Container for push operation results with metadata"""
    repo_id: str
    model_path: str
    success: bool
    timestamp: str
    commit_url: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def save(self, filepath: Path) -> None:
        """Save results to JSON file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ValidationResult:
    """Container for validation check results"""
    check_name: str
    passed: bool
    message: str
    is_warning: bool = False
    
    def __str__(self) -> str:
        status = "[OK]" if self.passed else ("[WARN]" if self.is_warning else "[FAIL]")
        return f"{status} {self.check_name}: {self.message}"