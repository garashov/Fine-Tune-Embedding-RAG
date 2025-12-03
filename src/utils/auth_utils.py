import os
import logging
from typing import Optional
from huggingface_hub import HfApi, login, whoami

from src.utils.data_models import ValidationResult
from src.config.config import (
    HF_PUSH_AUTH_USE_AUTH_TOKEN,
    HF_PUSH_AUTH_TOKEN_SOURCE,
    HF_PUSH_AUTH_TOKEN_ENV_VAR,
)


class AuthenticationManager:
    """
    Manages HuggingFace authentication
    Can be moved to utils/auth_utils.py
    """
    
    def __init__(
        self,
        use_auth_token: bool = HF_PUSH_AUTH_USE_AUTH_TOKEN,
        token_source: str = HF_PUSH_AUTH_TOKEN_SOURCE,
        token_env_var: str = HF_PUSH_AUTH_TOKEN_ENV_VAR,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize authentication manager"""
        self.use_auth_token = use_auth_token
        self.token_source = token_source
        self.token_env_var = token_env_var
        self.logger = logger or logging.getLogger(__name__)
        self.api = HfApi()
    
    def get_token(self) -> Optional[str]:
        """
        Get authentication token based on configuration
        
        Returns:
            Token string or None
        """
        if not self.use_auth_token:
            return None
        
        if self.token_source == "env":
            token = os.getenv(self.token_env_var)
            if not token:
                raise ValueError(
                    f"Token source set to 'env' but {self.token_env_var} not found"
                )
            return token
        
        # Default to CLI token
        return True  # Will use token from huggingface-cli login
    
    def validate_authentication(self) -> ValidationResult:
        """
        Validate that user is authenticated
        
        Returns:
            ValidationResult with authentication status
        """
        try:
            user_info = whoami()
            username = user_info.get("name", "unknown")
            self.logger.info(f"Authenticated as: {username}")
            return ValidationResult(
                check_name="authentication",
                passed=True,
                message=f"Authenticated as {username}"
            )
        except Exception as e:
            return ValidationResult(
                check_name="authentication",
                passed=False,
                message=f"Not authenticated: {str(e)}. Run 'huggingface-cli login'"
            )

