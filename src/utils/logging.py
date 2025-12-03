import logging
from pathlib import Path
from typing import Optional

from src.config.config import (
    HF_PUSH_OUTPUT_LOGGING_LEVEL,
    HF_PUSH_OUTPUT_LOGGING_SAVE_TO_FILE,
    HF_PUSH_OUTPUT_LOGGING_CONSOLE_FORMAT,
    HF_PUSH_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT,
    HF_PUSH_OUTPUT_LOGGING_FILE_FORMAT,
    EVAL_OUTPUT_LOGGING_LEVEL,
    EVAL_OUTPUT_LOGGING_SAVE_TO_FILE,
    EVAL_OUTPUT_LOGGING_CONSOLE_FORMAT,
    EVAL_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT,
    EVAL_OUTPUT_LOGGING_FILE_FORMAT,
)


class EvaluationLogger:
    """
    Handles logging configuration for evaluation
    Can be moved to utils/logging.py
    """
    
    def __init__(
        self,
        name: str = "qwen3_evaluation",
        log_level: str = EVAL_OUTPUT_LOGGING_LEVEL,
        log_file: Optional[Path] = None,
        console_format: str = EVAL_OUTPUT_LOGGING_CONSOLE_FORMAT,
        console_date_format: str = EVAL_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT,
        file_format: str = EVAL_OUTPUT_LOGGING_FILE_FORMAT
    ):
        """
        Initialize logger with configuration
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional path to log file
            console_format: Format string for console output
            console_date_format: Date format for console output
            file_format: Format string for file output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()
        
        # Console handler
        self._setup_console_handler(console_format, console_date_format)
        
        # File handler
        if log_file and EVAL_OUTPUT_LOGGING_SAVE_TO_FILE:
            self._setup_file_handler(log_file, file_format)
    
    def _setup_console_handler(self, format_str: str, date_format: str) -> None:
        """Setup console logging handler"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(format_str, datefmt=date_format)
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: Path, format_str: str) -> None:
        """Setup file logging handler"""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(format_str)
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger




class HubPushLogger:
    """
    Handles logging configuration for hub push operations
    Can be moved to utils/logging.py
    """
    
    def __init__(
        self,
        name: str = "qwen3_hub_push",
        log_level: str = HF_PUSH_OUTPUT_LOGGING_LEVEL,
        log_file: Optional[Path] = None,
        console_format: str = HF_PUSH_OUTPUT_LOGGING_CONSOLE_FORMAT,
        console_date_format: str = HF_PUSH_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT,
        file_format: str = HF_PUSH_OUTPUT_LOGGING_FILE_FORMAT
    ):
        """Initialize logger with configuration"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()
        
        # Console handler
        self._setup_console_handler(console_format, console_date_format)
        
        # File handler
        if log_file and HF_PUSH_OUTPUT_LOGGING_SAVE_TO_FILE:
            self._setup_file_handler(log_file, file_format)
    
    def _setup_console_handler(self, format_str: str, date_format: str) -> None:
        """Setup console logging handler"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(format_str, datefmt=date_format)
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: Path, format_str: str) -> None:
        """Setup file logging handler"""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(format_str)
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger
