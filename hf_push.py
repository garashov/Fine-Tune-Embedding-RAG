"""
Push Fine-tuned Qwen3-Embedding Model to HuggingFace Hub

Usage: Configure the variables in the CONFIG section and run the script
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from sentence_transformers import SentenceTransformer


# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to your fine-tuned model (REQUIRED)
MODEL_PATH = "data/fine_tuning/20251203_114114/finetuned_qwen3_embedding"

# HuggingFace Hub configuration
HF_REPO_ID = "elnurgar/qwen3-embedding-finetuned"  # Your HuggingFace repo ID (username/model-name)
HF_PRIVATE = False  # Set to True to make the repository private
HF_COMMIT_MESSAGE = None  # Custom commit message (None for auto-generated timestamp)

# Additional Hub options
HF_CREATE_PR = False  # Create a Pull Request instead of direct push
HF_REVISION = None  # Target branch/revision (None = main/default)

# Output directory for logs
LOG_DIR = Path("data/logs")


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging for hub push
    
    Args:
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("qwen3_hub_push")
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
# Hub Push Function
# ============================================================================
def push_to_hub(
    model_path: str,
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    revision: Optional[str] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Push model to HuggingFace Hub
    
    Args:
        model_path: Path to the fine-tuned model
        repo_id: Repository ID on HuggingFace Hub (username/model-name)
        commit_message: Optional commit message
        private: Whether to make the repo private
        create_pr: Whether to create a Pull Request
        revision: Target branch/revision
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("PUSHING MODEL TO HUGGINGFACE HUB")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Repository: {repo_id}")
    logger.info(f"  Private: {private}")
    logger.info(f"  Create PR: {create_pr}")
    if revision:
        logger.info(f"  Revision: {revision}")
    
    # Validate model path
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    logger.info("\n" + "-" * 80)
    logger.info("STEP 1: Loading Model")
    logger.info("-" * 80)
    
    try:
        # Load model
        logger.info("Loading model from local path...")
        model = SentenceTransformer(model_path)
        logger.info("✓ Model loaded successfully")
        
        # Set default commit message if not provided
        if commit_message is None:
            commit_message = f"Upload fine-tuned Qwen3-Embedding model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: Uploading to HuggingFace Hub")
        logger.info("-" * 80)
        logger.info(f"Commit message: {commit_message}")
        logger.info("This may take a few minutes depending on model size and connection speed...")
        
        # Push to hub
        model.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            create_pr=create_pr,
            revision=revision
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ UPLOAD SUCCESSFUL")
        logger.info("=" * 80)
        
        # Provide usage information
        if create_pr:
            logger.info(f"\nPull Request created at: https://huggingface.co/{repo_id}")
            logger.info("Review and merge the PR to make the model available")
        else:
            logger.info(f"\nModel available at: https://huggingface.co/{repo_id}")
            logger.info("\nTo use your model:")
            logger.info("  ```python")
            logger.info("  from sentence_transformers import SentenceTransformer")
            logger.info(f"  model = SentenceTransformer('{repo_id}')")
            logger.info("  ```")
            logger.info("\nTo use in your code:")
            logger.info("  ```python")
            logger.info("  # Generate embeddings")
            logger.info("  sentences = ['Example sentence 1', 'Example sentence 2']")
            logger.info("  embeddings = model.encode(sentences)")
            logger.info("  ```")
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("✗ UPLOAD FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("\nCommon issues:")
        logger.error("  - Not logged in to HuggingFace (run: huggingface-cli login)")
        logger.error("  - Invalid repository ID format (should be: username/model-name)")
        logger.error("  - Network connection issues")
        logger.error("  - Insufficient permissions for the repository")
        raise


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main push pipeline"""
    
    # Validate configuration
    if MODEL_PATH is None:
        raise ValueError("MODEL_PATH must be set")
    
    if HF_REPO_ID is None:
        raise ValueError("HF_REPO_ID must be set (format: username/model-name)")
    
    # Validate repo_id format
    if "/" not in HF_REPO_ID:
        raise ValueError(
            f"Invalid HF_REPO_ID format: '{HF_REPO_ID}'. "
            "Must be in format 'username/model-name'"
        )
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"push_to_hub_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("QWEN3-EMBEDDING HUB PUSH")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    
    try:
        push_to_hub(
            model_path=MODEL_PATH,
            repo_id=HF_REPO_ID,
            commit_message=HF_COMMIT_MESSAGE,
            private=HF_PRIVATE,
            create_pr=HF_CREATE_PR,
            revision=HF_REVISION,
            logger=logger
        )
        
        logger.info(f"\n✓ All operations completed successfully")
        logger.info(f"Log saved to: {log_file}")
        
    except Exception as e:
        logger.error(f"\n✗ Push failed: {str(e)}")
        logger.error(f"Full log available at: {log_file}")
        raise


if __name__ == "__main__":
    main()