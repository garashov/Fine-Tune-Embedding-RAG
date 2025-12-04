import os
import gc
import torch
import logging
import platform
from typing import Dict, Optional

from src.config.config import (
    FT_HARDWARE_CUDA_VISIBLE_DEVICES,
    FT_HARDWARE_NUM_THREADS,
)


class HardwareManager:
    """
    Manages hardware configuration and monitoring
    Can be moved to utils/hardware_utils.py
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize hardware manager"""
        self.logger = logger or logging.getLogger(__name__)
        self._configure_hardware()
    
    def _configure_hardware(self) -> None:
        """Configure hardware settings"""
        # Set CUDA visible devices
        if FT_HARDWARE_CUDA_VISIBLE_DEVICES:
            os.environ['CUDA_VISIBLE_DEVICES'] = FT_HARDWARE_CUDA_VISIBLE_DEVICES
            self.logger.info(f"CUDA_VISIBLE_DEVICES set to: {FT_HARDWARE_CUDA_VISIBLE_DEVICES}")
        
        # Set number of threads
        if FT_HARDWARE_NUM_THREADS:
            torch.set_num_threads(FT_HARDWARE_NUM_THREADS)
            self.logger.info(f"PyTorch threads set to: {FT_HARDWARE_NUM_THREADS}")
    
    def log_hardware_info(self) -> None:
        """Log hardware information"""
        self.logger.info("=" * 80)
        self.logger.info("HARDWARE INFORMATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Platform: {platform.system()} {platform.release()}")
        self.logger.info(f"Python version: {platform.python_version()}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"GPU {i}: {props.name}")
                self.logger.info(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                self.logger.info(f"  Compute capability: {props.major}.{props.minor}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    
    def clear_cache(self) -> None:
        """Clear CUDA cache and run garbage collection"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

