import logging
from pathlib import Path
from typing import Optional, List
from sentence_transformers import SentenceTransformer

from src.utils.auth_utils import AuthenticationManager
from src.utils.data_models import ValidationResult
from src.config.config import (
    HF_PUSH_VALIDATION_ENABLED,
    HF_PUSH_VALIDATION_CHECKS,
    HF_PUSH_VALIDATION_STRICT_MODE,
    HF_PUSH_VALIDATION_TIMEOUT,

)


class PushValidator:
    """
    Validates conditions before pushing to hub
    Can be moved to utils/validators.py
    """
    
    def __init__(
        self,
        enabled: bool = HF_PUSH_VALIDATION_ENABLED,
        checks: List[str] = HF_PUSH_VALIDATION_CHECKS,
        strict_mode: bool = HF_PUSH_VALIDATION_STRICT_MODE,
        timeout: int = HF_PUSH_VALIDATION_TIMEOUT,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize validator"""
        self.enabled = enabled
        self.checks = checks
        self.strict_mode = strict_mode
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self.auth_manager = AuthenticationManager(logger=logger)
    
    def validate_all(
        self,
        model_path: Path,
        repo_id: str
    ) -> List[ValidationResult]:
        """
        Run all configured validation checks
        
        Args:
            model_path: Path to model
            repo_id: Repository ID
        
        Returns:
            List of ValidationResult objects
        """
        if not self.enabled:
            self.logger.info("Validation disabled, skipping checks")
            return []
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VALIDATION CHECKS")
        self.logger.info("=" * 80)
        
        results = []
        
        if "model_exists" in self.checks:
            results.append(self._check_model_exists(model_path))
        
        if "model_loadable" in self.checks:
            results.append(self._check_model_loadable(model_path))
        
        if "repo_id_valid" in self.checks:
            results.append(self._check_repo_id_valid(repo_id))
        
        if "auth_valid" in self.checks:
            results.append(self.auth_manager.validate_authentication())
        
        # Log results
        for result in results:
            log_func = self.logger.info if result.passed else (
                self.logger.warning if result.is_warning else self.logger.error
            )
            log_func(str(result))
        
        # Check for failures
        failures = [r for r in results if not r.passed and not r.is_warning]
        warnings = [r for r in results if not r.passed and r.is_warning]
        
        if failures:
            raise ValueError(
                f"Validation failed: {len(failures)} check(s) failed. "
                "Fix issues before pushing."
            )
        
        if warnings and self.strict_mode:
            raise ValueError(
                f"Validation failed (strict mode): {len(warnings)} warning(s). "
                "Resolve warnings or disable strict mode."
            )
        
        self.logger.info("\n✓ All validation checks passed")
        return results
    
    def _check_model_exists(self, model_path: Path) -> ValidationResult:
        """Check if model path exists"""
        if model_path.exists():
            return ValidationResult(
                check_name="model_exists",
                passed=True,
                message=f"Model path found: {model_path}"
            )
        return ValidationResult(
            check_name="model_exists",
            passed=False,
            message=f"Model path not found: {model_path}"
        )
    
    def _check_model_loadable(self, model_path: Path) -> ValidationResult:
        """Check if model can be loaded"""
        try:
            self.logger.info("Testing model loading...")
            model = SentenceTransformer(str(model_path))
            del model  # Clean up
            return ValidationResult(
                check_name="model_loadable",
                passed=True,
                message="Model loaded successfully"
            )
        except Exception as e:
            return ValidationResult(
                check_name="model_loadable",
                passed=False,
                message=f"Failed to load model: {str(e)}"
            )
    
    def _check_repo_id_valid(self, repo_id: str) -> ValidationResult:
        """Check if repository ID format is valid"""
        if not repo_id:
            return ValidationResult(
                check_name="repo_id_valid",
                passed=False,
                message="Repository ID is empty"
            )
        
        if "/" not in repo_id:
            return ValidationResult(
                check_name="repo_id_valid",
                passed=False,
                message=f"Invalid format: '{repo_id}'. Must be 'username/model-name'"
            )
        
        parts = repo_id.split("/")
        if len(parts) != 2 or not all(parts):
            return ValidationResult(
                check_name="repo_id_valid",
                passed=False,
                message=f"Invalid format: '{repo_id}'. Must be 'username/model-name'"
            )
        
        return ValidationResult(
            check_name="repo_id_valid",
            passed=True,
            message=f"Valid repository ID: {repo_id}"
        )

