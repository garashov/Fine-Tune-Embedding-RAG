import logging
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset, concatenate_datasets, Dataset

from src.config.config import (
    FT_TRAINING_SEED,
    FT_DATASET_DATASET_NAME,
    FT_DATASET_DATASET_SPLIT,
    FT_DATASET_TEST_SIZE,
    FT_DATASET_ANCHOR_COLUMN,
    FT_DATASET_POSITIVE_COLUMN,
    FT_DATASET_SHUFFLE_TRAIN,
    FT_DATASET_SHUFFLE_SEED,
    FT_DATASET_MAX_TRAIN_SAMPLES,
    FT_DATASET_MAX_EVAL_SAMPLES,
    FT_DATASET_CACHE_DIR,
    FT_DATASET_KEEP_IN_MEMORY,
    FT_EVALUATION_SAVE_EVAL_DATA,
    FT_EVALUATION_EVAL_DATA_SUBDIR,
)


class DatasetManager:
    """
    Handles dataset loading and preparation
    Can be moved to core/dataset_manager.py
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize dataset manager"""
        self.logger = logger or logging.getLogger(__name__)
    
    def load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load and prepare the training dataset
        
        Returns:
            Tuple of (train_dataset, test_dataset, corpus_dataset)
        """
        self.logger.info("=" * 80)
        self.logger.info("LOADING DATASET")
        self.logger.info("=" * 80)
        self.logger.info(f"Dataset: {FT_DATASET_DATASET_NAME}")
        self.logger.info(f"Split: {FT_DATASET_DATASET_SPLIT}")
        
        try:
            # Load dataset
            dataset = load_dataset(
                FT_DATASET_DATASET_NAME,
                split=FT_DATASET_DATASET_SPLIT,
                cache_dir=FT_DATASET_CACHE_DIR,
                keep_in_memory=FT_DATASET_KEEP_IN_MEMORY
            )
            
            self.logger.info(f"[OK] Loaded {len(dataset)} examples")
            
            # Rename columns
            dataset = dataset.rename_column(FT_DATASET_ANCHOR_COLUMN, "anchor")
            dataset = dataset.rename_column(FT_DATASET_POSITIVE_COLUMN, "positive")
            
            # Add ID column
            dataset = dataset.add_column("id", range(len(dataset)))
            
            # Shuffle if configured
            if FT_DATASET_SHUFFLE_TRAIN:
                dataset = dataset.shuffle(seed=FT_DATASET_SHUFFLE_SEED)
                self.logger.info(f"[OK] Dataset shuffled (seed={FT_DATASET_SHUFFLE_SEED})")
            
            # Split into train and test
            split_dataset = dataset.train_test_split(
                test_size=FT_DATASET_TEST_SIZE,
                seed=FT_TRAINING_SEED
            )
            
            train_dataset = split_dataset["train"]
            test_dataset = split_dataset["test"]
            
            # Limit samples if configured
            if FT_DATASET_MAX_TRAIN_SAMPLES:
                train_dataset = train_dataset.select(range(min(len(train_dataset), FT_DATASET_MAX_TRAIN_SAMPLES)))
                self.logger.info(f"Limited train set to {len(train_dataset)} samples")
            
            if FT_DATASET_MAX_EVAL_SAMPLES:
                test_dataset = test_dataset.select(range(min(len(test_dataset), FT_DATASET_MAX_EVAL_SAMPLES)))
                self.logger.info(f"Limited test set to {len(test_dataset)} samples")
            
            self.logger.info(f"[OK] Train set: {len(train_dataset)} examples")
            self.logger.info(f"[OK] Test set: {len(test_dataset)} examples")
            
            # Create full corpus for evaluation
            corpus_dataset = concatenate_datasets([train_dataset, test_dataset])
            
            return train_dataset, test_dataset, corpus_dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}", exc_info=True)
            raise
    
    def save_evaluation_data(
        self,
        output_dir: Path,
        test_dataset: Dataset,
        corpus_dataset: Dataset
    ) -> None:
        """
        Save test and corpus data for later evaluation
        
        Args:
            output_dir: Output directory
            test_dataset: Test dataset
            corpus_dataset: Full corpus dataset
        """
        if not FT_EVALUATION_SAVE_EVAL_DATA:
            return
        
        eval_data_dir = output_dir / FT_EVALUATION_EVAL_DATA_SUBDIR
        eval_data_dir.mkdir(parents=True, exist_ok=True)
        
        test_dataset.save_to_disk(str(eval_data_dir / "test_dataset"))
        corpus_dataset.save_to_disk(str(eval_data_dir / "corpus_dataset"))
        
        self.logger.info(f"[OK] Evaluation data saved to: {eval_data_dir}")

