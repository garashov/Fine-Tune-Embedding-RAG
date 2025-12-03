import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import time
from sentence_transformers import SentenceTransformer

from src.utils.data_models import PushResult
from src.config.config import (
    HF_PUSH_UPLOAD_MAX_RETRIES,
    HF_PUSH_UPLOAD_RETRY_DELAY_SECONDS,
    HF_PUSH_UPLOAD_TIMEOUT,
)


class HubPusher:
    """
    Handles pushing models to HuggingFace Hub
    Can be moved to core/hub_pusher.py
    """
    
    def __init__(
        self,
        max_retries: int = HF_PUSH_UPLOAD_MAX_RETRIES,
        retry_delay: int = HF_PUSH_UPLOAD_RETRY_DELAY_SECONDS,
        timeout: int = HF_PUSH_UPLOAD_TIMEOUT,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize hub pusher"""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
    
    def push(
        self,
        model_path: Path,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: bool = False,
        create_pr: bool = False,
        revision: Optional[str] = None,
        model_card_content: Optional[str] = None
    ) -> PushResult:
        """
        Push model to HuggingFace Hub with retry logic
        
        Args:
            model_path: Path to model
            repo_id: Repository ID
            commit_message: Commit message
            private: Make repo private
            create_pr: Create pull request
            revision: Target revision
            model_card_content: Optional model card content
        
        Returns:
            PushResult with operation details
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PUSHING TO HUGGINGFACE HUB")
        self.logger.info("=" * 80)
        self.logger.info(f"Repository: {repo_id}")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Private: {private}")
        self.logger.info(f"Create PR: {create_pr}")
        if revision:
            self.logger.info(f"Revision: {revision}")
        
        start_time = time.time()
        
        # Set default commit message
        if not commit_message:
            commit_message = (
                f"Upload fine-tuned Qwen3-Embedding model - "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        # Load model
        try:
            self.logger.info("\nLoading model...")
            model = SentenceTransformer(str(model_path))
            self.logger.info("✓ Model loaded successfully")
        except Exception as e:
            return PushResult(
                repo_id=repo_id,
                model_path=str(model_path),
                success=False,
                timestamp=datetime.now().isoformat(),
                error_message=f"Failed to load model: {str(e)}"
            )
        
        # Push with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"\nUpload attempt {attempt}/{self.max_retries}")
                self.logger.info(f"Commit message: {commit_message}")
                self.logger.info("Uploading... (this may take several minutes)")
                
                # Push to hub
                model.push_to_hub(
                    repo_id=repo_id,
                    commit_message=commit_message,
                    private=private,
                    create_pr=create_pr,
                    revision=revision
                )
                
                # Save model card if provided
                if model_card_content:
                    self._save_model_card(model_path, model_card_content)
                
                duration = time.time() - start_time
                
                self.logger.info("\n" + "=" * 80)
                self.logger.info("✓ UPLOAD SUCCESSFUL")
                self.logger.info("=" * 80)
                
                commit_url = f"https://huggingface.co/{repo_id}"
                if create_pr:
                    self.logger.info(f"\nPull Request: {commit_url}")
                else:
                    self.logger.info(f"\nModel URL: {commit_url}")
                
                self._log_usage_instructions(repo_id, create_pr)
                
                return PushResult(
                    repo_id=repo_id,
                    model_path=str(model_path),
                    success=True,
                    timestamp=datetime.now().isoformat(),
                    commit_url=commit_url,
                    duration_seconds=duration
                )
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    duration = time.time() - start_time
                    self.logger.error("\n" + "=" * 80)
                    self.logger.error("✗ UPLOAD FAILED")
                    self.logger.error("=" * 80)
                    self._log_common_issues()
                    
                    return PushResult(
                        repo_id=repo_id,
                        model_path=str(model_path),
                        success=False,
                        timestamp=datetime.now().isoformat(),
                        error_message=str(e),
                        duration_seconds=duration
                    )
    
    def _save_model_card(self, model_path: Path, content: str) -> None:
        """Save model card to model directory"""
        try:
            card_path = model_path / "README.md"
            with open(card_path, 'w') as f:
                f.write(content)
            self.logger.info(f"✓ Model card saved: {card_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save model card: {str(e)}")
    
    def _log_usage_instructions(self, repo_id: str, is_pr: bool) -> None:
        """Log usage instructions for the pushed model"""
        if is_pr:
            self.logger.info("\nReview and merge the PR to make the model available")
        else:
            self.logger.info("\nTo use your model:")
            self.logger.info("```python")
            self.logger.info("from sentence_transformers import SentenceTransformer")
            self.logger.info(f"model = SentenceTransformer('{repo_id}')")
            self.logger.info("embeddings = model.encode(['Your text here'])")
            self.logger.info("```")
    
    def _log_common_issues(self) -> None:
        """Log common issues and solutions"""
        self.logger.error("\nCommon issues:")
        self.logger.error("  • Not authenticated (run: huggingface-cli login)")
        self.logger.error("  • Invalid repository ID format (should be: username/model-name)")
        self.logger.error("  • Network connection issues")
        self.logger.error("  • Insufficient permissions for the repository")
        self.logger.error("  • Model files too large (consider using Git LFS)")

