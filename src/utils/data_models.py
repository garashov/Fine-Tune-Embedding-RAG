import json
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, asdict

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
        status = "✓" if self.passed else ("⚠" if self.is_warning else "✗")
        return f"{status} {self.check_name}: {self.message}"


# ============================================================================
# Evalaution Data Models
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