import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi

from src.config.config import (
    HF_PUSH_POST_PUSH_VERIFY_UPLOAD_ENABLED,
    HF_PUSH_POST_PUSH_VERIFY_UPLOAD_TEST_DOWNLOAD,
    HF_PUSH_POST_PUSH_VERIFY_UPLOAD_VERIFY_CHECKSUMS,
)


class PostPushVerifier:
    """
    Verifies successful push operation
    Can be moved to utils/verifiers.py
    """
    
    def __init__(
        self,
        enabled: bool = HF_PUSH_POST_PUSH_VERIFY_UPLOAD_ENABLED,
        test_download: bool = HF_PUSH_POST_PUSH_VERIFY_UPLOAD_TEST_DOWNLOAD,
        verify_checksums: bool = HF_PUSH_POST_PUSH_VERIFY_UPLOAD_VERIFY_CHECKSUMS,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize verifier"""
        self.enabled = enabled
        self.test_download = test_download
        self.verify_checksums = verify_checksums
        self.logger = logger or logging.getLogger(__name__)
        self.api = HfApi()
    
    def verify(self, repo_id: str) -> bool:
        """
        Verify push was successful
        
        Args:
            repo_id: Repository ID to verify
        
        Returns:
            True if verification successful
        """
        if not self.enabled:
            self.logger.info("Post-push verification disabled")
            return True
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("POST-PUSH VERIFICATION")
        self.logger.info("-" * 80)
        
        try:
            # Check if repo exists
            self.logger.info(f"Checking repository: {repo_id}")
            repo_info = self.api.repo_info(repo_id)
            self.logger.info(f"✓ Repository found: {repo_info.id}")
            
            # Test download if enabled
            if self.test_download:
                self.logger.info("Testing model download...")
                model = SentenceTransformer(repo_id)
                self.logger.info("✓ Model downloaded and loaded successfully")
                del model
            
            self.logger.info("✓ Verification complete")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Verification failed: {str(e)}")
            return False

