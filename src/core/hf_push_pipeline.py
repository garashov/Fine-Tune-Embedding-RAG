"""
Production-Grade HuggingFace Hub Push Script for Qwen3-Embedding Models

This module provides a robust framework for pushing fine-tuned models to HuggingFace Hub with:
- Configuration-driven architecture
- Class-based design for reusability
- Comprehensive validation and error handling
- Model card generation
- Backup and verification capabilities
- Modular components ready for extraction to utils
"""
import json
from pathlib import Path
from datetime import datetime

from src.utils.hub_pusher import HubPusher
from src.utils.model_card import ModelCardGenerator
from src.utils.verifiers import PostPushVerifier
from src.utils.validators import PushValidator
from src.utils.backup_utils import BackupManager
from src.utils.logging import HubPushLogger
from src.utils.data_models import PushResult
from src.config.config import (
    HF_PUSH_MODEL_FT_PATH,
    HF_PUSH_REPO_ID,
    HF_PUSH_PRIVATE,
    HF_PUSH_COMMIT_MESSAGE,
    HF_PUSH_CREATE_PR,
    HF_PUSH_REVISION,
    HF_PUSH_MODEL_CARD_GENERATE_CARD,
    HF_PUSH_OUTPUT_LOGGING_SAVE_TO_FILE,
    HF_PUSH_OUTPUT_LOGGING_LOG_DIR,
    HF_PUSH_OUTPUT_LOGGING_LOG_FILENAME,
    HF_PUSH_VALIDATION_ENABLED,
    HF_PUSH_BACKUP_CREATE_BACKUP,
    HF_PUSH_POST_PUSH_ACTIONS,
)



# ============================================================================
# Main Push Pipeline
# ============================================================================

class PushPipeline:
    """
    Main pipeline orchestrating the entire push process
    """
    
    def __init__(self):
        """Initialize push pipeline with configuration"""
        self._validate_configuration()
        self._setup_paths()
        self._setup_logging()
        
        # Initialize components
        self.validator = PushValidator(logger=self.logger)
        self.backup_manager = BackupManager(logger=self.logger)
        self.model_card_generator = ModelCardGenerator(logger=self.logger)
        self.hub_pusher = HubPusher(logger=self.logger)
        self.verifier = PostPushVerifier(logger=self.logger)
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings"""
        if not HF_PUSH_MODEL_FT_PATH:
            raise ValueError("HF_PUSH_MODEL_FT_PATH must be set")
        
        if not HF_PUSH_REPO_ID:
            raise ValueError("HF_PUSH_REPO_ID must be set (format: username/model-name)")
    
    def _setup_paths(self) -> None:
        """Setup file paths"""
        self.model_path = Path(HF_PUSH_MODEL_FT_PATH)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = None
        if HF_PUSH_OUTPUT_LOGGING_SAVE_TO_FILE:
            log_dir = Path(HF_PUSH_OUTPUT_LOGGING_LOG_DIR)
            log_filename = HF_PUSH_OUTPUT_LOGGING_LOG_FILENAME.format(
                timestamp=self.timestamp
            )
            log_file = log_dir / log_filename
        
        logger_wrapper = HubPushLogger(log_file=log_file)
        self.logger = logger_wrapper.get_logger()
        
        self._log_configuration()
    
    def _log_configuration(self) -> None:
        """Log current configuration"""
        self.logger.info("=" * 80)
        self.logger.info("QWEN3-EMBEDDING HUB PUSH PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("\nConfiguration:")
        self.logger.info(f"  Model Path:       {HF_PUSH_MODEL_FT_PATH}")
        self.logger.info(f"  Repository ID:    {HF_PUSH_REPO_ID}")
        self.logger.info(f"  Private:          {HF_PUSH_PRIVATE}")
        self.logger.info(f"  Create PR:        {HF_PUSH_CREATE_PR}")
        self.logger.info(f"  Generate Card:    {HF_PUSH_MODEL_CARD_GENERATE_CARD}")
        self.logger.info(f"  Create Backup:    {HF_PUSH_BACKUP_CREATE_BACKUP}")
        self.logger.info(f"  Validation:       {HF_PUSH_VALIDATION_ENABLED}")
    
    def run(self) -> PushResult:
        """
        Execute the complete push pipeline
        
        Returns:
            PushResult with operation details
        """
        try:
            # Step 1: Validation
            self.validator.validate_all(
                model_path=self.model_path,
                repo_id=HF_PUSH_REPO_ID
            )
            
            # Step 2: Backup
            backup_path = self.backup_manager.backup_model(self.model_path)
            
            # Step 3: Generate model card
            model_card_content = self.model_card_generator.generate(
                model_path=self.model_path,
                repo_id=HF_PUSH_REPO_ID
            )
            
            # Step 4: Push to hub
            result = self.hub_pusher.push(
                model_path=self.model_path,
                repo_id=HF_PUSH_REPO_ID,
                commit_message=HF_PUSH_COMMIT_MESSAGE,
                private=HF_PUSH_PRIVATE,
                create_pr=HF_PUSH_CREATE_PR,
                revision=HF_PUSH_REVISION,
                model_card_content=model_card_content
            )
            
            # Step 5: Post-push verification
            if result.success:
                self.verifier.verify(HF_PUSH_REPO_ID)
            
            # Step 6: Save result
            self._save_push_history(result)
            
            # Log completion
            self._log_completion(result)
            
            return result
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("PUSH PIPELINE FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            
            result = PushResult(
                repo_id=HF_PUSH_REPO_ID,
                model_path=str(self.model_path),
                success=False,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            self._save_push_history(result)
            raise
    
    def _save_push_history(self, result: PushResult) -> None:
        """Save push result to history"""
        if HF_PUSH_POST_PUSH_ACTIONS.get("create_manifest", False):
            manifest_path = Path(HF_PUSH_POST_PUSH_ACTIONS.get(
                "manifest_path",
                "data/push_history.json"
            ))
            
            # Load existing history
            history = []
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    history = json.load(f)
            
            # Add new result
            history.append(result.to_dict())
            
            # Save updated history
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info(f"\nPush history updated: {manifest_path}")
    
    def _log_completion(self, result: PushResult) -> None:
        """Log completion message"""
        self.logger.info("=" * 80)
        if result.success:
            self.logger.info("[OK] PUSH PIPELINE COMPLETED SUCCESSFULLY")
        else:
            self.logger.info("[FAIL] PUSH PIPELINE FAILED")
        self.logger.info("=" * 80)
        
        if result.duration_seconds:
            self.logger.info(f"Total duration: {result.duration_seconds:.2f} seconds")
