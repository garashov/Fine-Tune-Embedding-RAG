import json
import torch
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config.config import (
    FT_MODEL_BASE_MODEL_ID,
    FT_MODEL_MAX_SEQ_LENGTH,
    FT_MODEL_LOAD_IN_4BIT,
    FT_LORA_R,
    FT_LORA_ALPHA,
    FT_LORA_DROPOUT,
    FT_LORA_TARGET_MODULES,
    FT_TRAINING_NUM_TRAIN_EPOCHS,
    FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE,
    FT_TRAINING_LEARNING_RATE,
    FT_TRAINING_OPTIMIZER,
    FT_DATASET_DATASET_NAME,
    FT_DATASET_TEST_SIZE,
    FT_LOSS_TYPE,
    FT_OUTPUT_SAVE_CONFIG,
    FT_OUTPUT_CONFIG_FILENAME,
)


class ConfigurationManager:
    """
    Manages configuration saving and loading
    Can be moved to utils/config_manager.py
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize configuration manager"""
        self.logger = logger or logging.getLogger(__name__)
    
    def save_config(self, output_dir: Path) -> None:
        """
        Save training configuration
        
        Args:
            output_dir: Output directory
        """
        if not FT_OUTPUT_SAVE_CONFIG:
            return
        
        config_path = output_dir / FT_OUTPUT_CONFIG_FILENAME
        
        config_dict = {
            "model": {
                "base_model_id": FT_MODEL_BASE_MODEL_ID,
                "max_seq_length": FT_MODEL_MAX_SEQ_LENGTH,
                "load_in_4bit": FT_MODEL_LOAD_IN_4BIT,
            },
            "lora": {
                "r": FT_LORA_R,
                "lora_alpha": FT_LORA_ALPHA,
                "lora_dropout": FT_LORA_DROPOUT,
                "target_modules": FT_LORA_TARGET_MODULES,
            },
            "training": {
                "num_train_epochs": FT_TRAINING_NUM_TRAIN_EPOCHS,
                "per_device_train_batch_size": FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE,
                "learning_rate": FT_TRAINING_LEARNING_RATE,
                "optimizer": FT_TRAINING_OPTIMIZER,
            },
            "dataset": {
                "dataset_name": FT_DATASET_DATASET_NAME,
                "test_size": FT_DATASET_TEST_SIZE,
            },
            "loss": {
                "type": FT_LOSS_TYPE,
            },
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"[OK] Configuration saved to {config_path}")
