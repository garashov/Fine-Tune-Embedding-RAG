import logging
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk

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
        Load and prepare the training dataset from HuggingFace Hub or local path
        
        Returns:
            Tuple of (train_dataset, test_dataset, corpus_dataset)
        """
        self.logger.info("=" * 80)
        self.logger.info("LOADING DATASET")
        self.logger.info("=" * 80)
        
        dataset_source = FT_DATASET_DATASET_NAME
        
        try:
            # Check if it's a local path
            source_path = Path(dataset_source)
            
            if source_path.exists():
                self.logger.info(f"Source: Local path")
                self.logger.info(f"Path: {source_path.resolve()}")
                
                # Check if it's a save_to_disk format (has dataset_info.json or state.json)
                is_dataset_dict = (source_path / "dataset_dict.json").exists()
                is_dataset = (source_path / "dataset_info.json").exists() or (source_path / "state.json").exists()
                
                if is_dataset_dict:
                    # It's a DatasetDict saved with save_to_disk()
                    self.logger.info("Format: HuggingFace DatasetDict (load_from_disk)")
                    dataset_dict = load_from_disk(str(source_path))
                    
                    # Extract the requested split
                    if FT_DATASET_DATASET_SPLIT and FT_DATASET_DATASET_SPLIT in dataset_dict:
                        dataset = dataset_dict[FT_DATASET_DATASET_SPLIT]
                        self.logger.info(f"Using split: {FT_DATASET_DATASET_SPLIT}")
                    else:
                        available_splits = list(dataset_dict.keys())
                        self.logger.warning(f"Split '{FT_DATASET_DATASET_SPLIT}' not found. Available: {available_splits}")
                        dataset = dataset_dict[available_splits[0]]
                        self.logger.info(f"Using first available split: {available_splits[0]}")
                
                elif is_dataset:
                    # It's a single Dataset saved with save_to_disk()
                    self.logger.info("Format: HuggingFace Dataset (load_from_disk)")
                    dataset = load_from_disk(str(source_path))
                
                else:
                    # Let HF auto-detect format (CSV, JSON, Parquet, etc.)
                    self.logger.info("Format: Auto-detected by HuggingFace")
                    dataset = load_dataset(
                        str(source_path),
                        split=FT_DATASET_DATASET_SPLIT,
                        cache_dir=FT_DATASET_CACHE_DIR,
                        keep_in_memory=FT_DATASET_KEEP_IN_MEMORY
                    )
            else:
                self.logger.info(f"Source: HuggingFace Hub")
                self.logger.info(f"Dataset: {dataset_source}")
                self.logger.info(f"Split: {FT_DATASET_DATASET_SPLIT}")
                
                dataset = load_dataset(
                    dataset_source,
                    split=FT_DATASET_DATASET_SPLIT,
                    cache_dir=FT_DATASET_CACHE_DIR,
                    keep_in_memory=FT_DATASET_KEEP_IN_MEMORY
                )
        
            self.logger.info(f"[OK] Loaded {len(dataset)} examples")
            self.logger.info(f"Columns: {dataset.column_names}")
            
            # Rename columns only if needed
            if FT_DATASET_ANCHOR_COLUMN != "anchor" and FT_DATASET_ANCHOR_COLUMN in dataset.column_names:
                dataset = dataset.rename_column(FT_DATASET_ANCHOR_COLUMN, "anchor")
                self.logger.info(f"Renamed '{FT_DATASET_ANCHOR_COLUMN}' -> 'anchor'")

            if FT_DATASET_POSITIVE_COLUMN != "positive" and FT_DATASET_POSITIVE_COLUMN in dataset.column_names:
                dataset = dataset.rename_column(FT_DATASET_POSITIVE_COLUMN, "positive")
                self.logger.info(f"Renamed '{FT_DATASET_POSITIVE_COLUMN}' -> 'positive'")
                
            # Validate required columns exist
            if "anchor" not in dataset.column_names or "positive" not in dataset.column_names:
                raise ValueError(
                    f"Dataset must contain 'anchor' and 'positive' columns. "
                    f"Found: {dataset.column_names}. "
                    f"Check your anchor_column ('{FT_DATASET_ANCHOR_COLUMN}') and "
                    f"positive_column ('{FT_DATASET_POSITIVE_COLUMN}') settings."
                )
            
            # Add ID column only if it doesn't exist
            if "id" not in dataset.column_names:
                dataset = dataset.add_column("id", range(len(dataset)))
                self.logger.info("Added 'id' column")
            else:
                self.logger.info("'id' column already exists, skipping")
            
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


# ----------------------------
# Example of usage
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DatasetManagerExample")
    
    dataset_manager = DatasetManager(logger=logger)
    
    train_ds, test_ds, corpus_ds = dataset_manager.load_dataset()