import gc
import torch
from typing import  Optional



class GPUMemoryManager:
    """
    Manages GPU memory operations and reporting
    Can be moved to utils/gpu_utils.py
    """
    
    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available"""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_memory_allocated_gb() -> float:
        """Get currently allocated GPU memory in GB"""
        if not GPUMemoryManager.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024**3
    
    @staticmethod
    def clear_cache() -> None:
        """Clear CUDA cache"""
        if GPUMemoryManager.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def force_garbage_collection() -> None:
        """Force Python garbage collection"""
        gc.collect()
    
    @staticmethod
    def cleanup(clear_cache: bool = True, force_gc: bool = True) -> None:
        """Perform comprehensive memory cleanup"""
        if force_gc:
            GPUMemoryManager.force_garbage_collection()
        if clear_cache and GPUMemoryManager.is_available():
            GPUMemoryManager.clear_cache()
    
    @staticmethod
    def get_device(preferred_device: Optional[str] = None) -> str:
        """
        Get appropriate device for computation
        
        Args:
            preferred_device: Preferred device ('cuda', 'cpu', or None for auto)
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if preferred_device:
            if preferred_device == "cuda" and not GPUMemoryManager.is_available():
                raise ValueError("CUDA requested but not available")
            return preferred_device
        return "cuda" if GPUMemoryManager.is_available() else "cpu"
