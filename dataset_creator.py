"""
Production-Grade Dataset Creation Script for Embedding Fine-Tuning

This module provides tools to create HuggingFace-compatible datasets from
various input formats (CSV, JSON, etc.) for use with the fine-tuning pipeline.

Author: Your Name
Date: 2025-12-05
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from datasets import Dataset, DatasetDict, Features, Value, load_dataset


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class DatasetStatistics:
    """Statistics about the created dataset"""
    total_samples: int
    train_samples: int
    test_samples: int
    validation_samples: int
    
    avg_anchor_length: float
    avg_positive_length: float
    
    duplicate_count: int
    duplicate_rate: float
    
    min_anchor_length: int
    max_anchor_length: int
    min_positive_length: int
    max_positive_length: int
    
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def save(self, filepath: Path) -> None:
        """Save statistics to JSON"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Logger Setup
# ============================================================================
class DatasetLogger:
    """Handles logging for dataset creation"""
    
    def __init__(
        self,
        name: str = "dataset_creator",
        log_level: str = "INFO",
        log_file: Optional[Path] = None
    ):
        """Initialize logger"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()
        self.logger.propagate = False
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get logger instance"""
        return self.logger


# ============================================================================
# Data Loader
# ============================================================================
class DataLoader:
    """Loads data from various file formats"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize data loader"""
        self.logger = logger or logging.getLogger(__name__)
    
    def load(
        self,
        file_path: Path,
        file_format: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            file_path: Path to input file
            file_format: File format (csv, json, jsonl, parquet)
            **kwargs: Additional arguments for pandas readers
        
        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Loading data from: {file_path}")
        self.logger.info(f"Format: {file_format}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if file_format == "csv":
                df = pd.read_csv(file_path, **kwargs)
            elif file_format == "json":
                df = pd.read_json(file_path, **kwargs)
            elif file_format == "jsonl":
                df = pd.read_json(file_path, lines=True, **kwargs)
            elif file_format == "parquet":
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            self.logger.info(f"Loaded {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise


# ============================================================================
# Data Processor
# ============================================================================
class DataProcessor:
    """Processes and cleans data"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize data processor"""
        self.logger = logger or logging.getLogger(__name__)
    
    def process(
        self,
        df: pd.DataFrame,
        anchor_col: str,
        positive_col: str,
        cleaning_config: Dict,
        validation_config: Dict
    ) -> pd.DataFrame:
        """
        Process and clean data
        
        Args:
            df: Input DataFrame
            anchor_col: Anchor column name
            positive_col: Positive column name
            cleaning_config: Cleaning configuration
            validation_config: Validation configuration
        
        Returns:
            Processed DataFrame
        """
        self.logger.info("Processing data...")
        
        # Rename columns
        df = df.rename(columns={
            anchor_col: "anchor",
            positive_col: "positive"
        })
        
        # Validate required columns
        if "anchor" not in df.columns or "positive" not in df.columns:
            raise ValueError("Missing required columns after renaming")
        
        initial_count = len(df)
        
        # Clean data
        if cleaning_config.get("strip_whitespace", True):
            df["anchor"] = df["anchor"].str.strip()
            df["positive"] = df["positive"].str.strip()
        
        if cleaning_config.get("normalize_spaces", True):
            df["anchor"] = df["anchor"].str.replace(r'\s+', ' ', regex=True)
            df["positive"] = df["positive"].str.replace(r'\s+', ' ', regex=True)
        
        if cleaning_config.get("remove_empty_lines", True):
            df = df.dropna(subset=["anchor", "positive"])
            df = df[df["anchor"].str.len() > 0]
            df = df[df["positive"].str.len() > 0]
        
        # Validate lengths
        min_anchor = validation_config.get("min_anchor_length", 3)
        max_anchor = validation_config.get("max_anchor_length", 512)
        min_positive = validation_config.get("min_positive_length", 10)
        max_positive = validation_config.get("max_positive_length", 2048)
        
        df = df[df["anchor"].str.len() >= min_anchor]
        df = df[df["anchor"].str.len() <= max_anchor]
        df = df[df["positive"].str.len() >= min_positive]
        df = df[df["positive"].str.len() <= max_positive]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} rows during cleaning/validation")
        
        # Add ID column
        df = df.reset_index(drop=True)
        df.insert(0, "id", range(len(df)))
        
        self.logger.info(f"Processing complete: {len(df)} rows remaining")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=["anchor", "positive"])
        duplicates = initial_count - len(df)
        
        if duplicates > 0:
            self.logger.info(f"Removed {duplicates} duplicate rows")
        
        return df, duplicates


# ============================================================================
# Dataset Creator
# ============================================================================
class DatasetCreator:
    """Creates HuggingFace datasets"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize dataset creator"""
        self.logger = logger or logging.getLogger(__name__)
    
    def create_splits(
        self,
        df: pd.DataFrame,
        test_size: float,
        validation_size: float,
        shuffle: bool,
        seed: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/test/validation
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            shuffle: Whether to shuffle
            seed: Random seed
        
        Returns:
            Dictionary with split DataFrames
        """
        self.logger.info("Creating data splits...")
        
        if shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        total_size = len(df)
        test_count = int(total_size * test_size)
        val_count = int(total_size * validation_size)
        train_count = total_size - test_count - val_count
        
        splits = {}
        
        # Create splits
        if validation_size > 0:
            splits["train"] = df.iloc[:train_count].reset_index(drop=True)
            splits["validation"] = df.iloc[train_count:train_count+val_count].reset_index(drop=True)
            splits["test"] = df.iloc[train_count+val_count:].reset_index(drop=True)
            
            self.logger.info(f"Train: {len(splits['train'])} samples")
            self.logger.info(f"Validation: {len(splits['validation'])} samples")
            self.logger.info(f"Test: {len(splits['test'])} samples")
        else:
            splits["train"] = df.iloc[:train_count].reset_index(drop=True)
            splits["test"] = df.iloc[train_count:].reset_index(drop=True)
            
            self.logger.info(f"Train: {len(splits['train'])} samples")
            self.logger.info(f"Test: {len(splits['test'])} samples")
        
        return splits
    
    def create_dataset_dict(
        self,
        splits: Dict[str, pd.DataFrame]
    ) -> DatasetDict:
        """
        Create HuggingFace DatasetDict
        
        Args:
            splits: Dictionary with split DataFrames
        
        Returns:
            HuggingFace DatasetDict
        """
        self.logger.info("Creating HuggingFace DatasetDict...")
        
        dataset_dict = {}
        for split_name, df in splits.items():
            dataset_dict[split_name] = Dataset.from_pandas(df, preserve_index=False)
        
        return DatasetDict(dataset_dict)
    
    def save_dataset(
        self,
        dataset: DatasetDict,
        output_dir: Path,
        dataset_name: str
    ) -> None:
        """
        Save dataset to disk
        
        Args:
            dataset: HuggingFace DatasetDict
            output_dir: Output directory
            dataset_name: Dataset name
        """
        save_path = output_dir / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving dataset to: {save_path}")
        dataset.save_to_disk(str(save_path))
        self.logger.info("Dataset saved successfully")


# ============================================================================
# Example Display
# ============================================================================
class ExampleDisplayer:
    """Displays example rows from dataset"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize example displayer"""
        self.logger = logger or logging.getLogger(__name__)
    
    def display_examples(
        self,
        splits: Dict[str, pd.DataFrame],
        num_examples: int = 3
    ) -> None:
        """
        Display example rows from each split
        
        Args:
            splits: Dictionary with split DataFrames
            num_examples: Number of examples to display per split
        """
        self.logger.info("=" * 80)
        self.logger.info("DATASET EXAMPLES")
        self.logger.info("=" * 80)
        
        for split_name, df in splits.items():
            self.logger.info(f"{split_name.upper()} SPLIT - Showing {min(num_examples, len(df))} examples:")
            self.logger.info("-" * 80)
            
            # Get random examples
            sample_df = df.sample(n=min(num_examples, len(df)), random_state=42)
            
            for idx, row in sample_df.iterrows():
                self.logger.info(f"Example {idx + 1}:")
                self.logger.info(f"  ID: {row['id']}")
                self.logger.info(f"  Anchor: {self._truncate(row['anchor'], 100)}")
                self.logger.info(f"  Positive: {self._truncate(row['positive'], 150)}")
                self.logger.info(f"  Lengths: anchor={len(row['anchor'])}, positive={len(row['positive'])}")
        
        self.logger.info("=" * 80)
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text for display"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def save_examples_to_file(
        self,
        splits: Dict[str, pd.DataFrame],
        output_dir: Path,
        num_examples: int = 5
    ) -> None:
        """
        Save examples to a text file for review
        
        Args:
            splits: Dictionary with split DataFrames
            output_dir: Output directory
            num_examples: Number of examples to save per split
        """
        examples_file = output_dir / "example_samples.txt"
        
        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET EXAMPLE SAMPLES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            
            for split_name, df in splits.items():
                f.write(f"\n\n{'=' * 80}\n")
                f.write(f"{split_name.upper()} SPLIT - {len(df)} total samples\n")
                f.write(f"Showing {min(num_examples, len(df))} random examples\n")
                f.write("=" * 80 + "\n")
                
                sample_df = df.sample(n=min(num_examples, len(df)), random_state=42)
                
                for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Example {i} (ID: {row['id']})\n")
                    f.write(f"{'-' * 80}\n")
                    f.write(f"ANCHOR ({len(row['anchor'])} chars):\n")
                    f.write(f"{row['anchor']}\n\n")
                    f.write(f"POSITIVE ({len(row['positive'])} chars):\n")
                    f.write(f"{row['positive']}\n")
        
        self.logger.info(f"Example samples saved to: {examples_file}")


# ============================================================================
# Statistics Calculator
# ============================================================================

class StatisticsCalculator:
    """Calculates dataset statistics"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize statistics calculator"""
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate(
        self,
        splits: Dict[str, pd.DataFrame],
        duplicate_count: int
    ) -> DatasetStatistics:
        """
        Calculate dataset statistics
        
        Args:
            splits: Dictionary with split DataFrames
            duplicate_count: Number of duplicates removed
        
        Returns:
            DatasetStatistics object
        """
        self.logger.info("Calculating statistics...")
        
        # Combine all splits
        all_data = pd.concat(splits.values(), ignore_index=True)
        
        total_samples = len(all_data)
        train_samples = len(splits.get("train", pd.DataFrame()))
        test_samples = len(splits.get("test", pd.DataFrame()))
        validation_samples = len(splits.get("validation", pd.DataFrame()))
        
        # Length statistics
        anchor_lengths = all_data["anchor"].str.len()
        positive_lengths = all_data["positive"].str.len()
        
        stats = DatasetStatistics(
            total_samples=total_samples,
            train_samples=train_samples,
            test_samples=test_samples,
            validation_samples=validation_samples,
            
            avg_anchor_length=float(anchor_lengths.mean()),
            avg_positive_length=float(positive_lengths.mean()),
            
            duplicate_count=duplicate_count,
            duplicate_rate=duplicate_count / (total_samples + duplicate_count) if total_samples > 0 else 0,
            
            min_anchor_length=int(anchor_lengths.min()),
            max_anchor_length=int(anchor_lengths.max()),
            min_positive_length=int(positive_lengths.min()),
            max_positive_length=int(positive_lengths.max()),
            
            timestamp=datetime.now().isoformat()
        )
        
        self._log_statistics(stats)
        
        return stats
    
    def _log_statistics(self, stats: DatasetStatistics) -> None:
        """Log statistics"""
        self.logger.info("=" * 80)
        self.logger.info("DATASET STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total samples: {stats.total_samples}")
        self.logger.info(f"  Train: {stats.train_samples}")
        self.logger.info(f"  Test: {stats.test_samples}")
        if stats.validation_samples > 0:
            self.logger.info(f"  Validation: {stats.validation_samples}")
        self.logger.info(f"Anchor (query) statistics:")
        self.logger.info(f"  Average length: {stats.avg_anchor_length:.1f} chars")
        self.logger.info(f"  Min length: {stats.min_anchor_length} chars")
        self.logger.info(f"  Max length: {stats.max_anchor_length} chars")
        self.logger.info(f"Positive (context) statistics:")
        self.logger.info(f"  Average length: {stats.avg_positive_length:.1f} chars")
        self.logger.info(f"  Min length: {stats.min_positive_length} chars")
        self.logger.info(f"  Max length: {stats.max_positive_length} chars")
        self.logger.info(f"Duplicates removed: {stats.duplicate_count} ({stats.duplicate_rate*100:.2f}%)")
        self.logger.info("=" * 80)


# ============================================================================
# Main Pipeline
# ============================================================================

class DatasetCreationPipeline:
    """Main pipeline for dataset creation"""
    
    def __init__(self, config: Dict):
        """Initialize pipeline with configuration"""
        self.config = config
        self._setup_logging()
        
        # Initialize components
        self.data_loader = DataLoader(logger=self.logger)
        self.data_processor = DataProcessor(logger=self.logger)
        self.dataset_creator = DatasetCreator(logger=self.logger)
        self.stats_calculator = StatisticsCalculator(logger=self.logger)
        self.example_displayer = ExampleDisplayer(logger=self.logger)
    
    def _setup_logging(self) -> None:
        """Setup logging"""
        log_config = self.config.get("logging", {})
        
        log_file = None
        if log_config.get("save_to_file", True):
            log_dir = Path(log_config.get("log_dir", "data/logs"))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = log_config.get("log_filename", "dataset_creation_{timestamp}.log")
            log_filename = log_filename.replace("{timestamp}", timestamp)
            log_file = log_dir / log_filename
        
        logger_wrapper = DatasetLogger(
            log_level=log_config.get("level", "INFO"),
            log_file=log_file
        )
        self.logger = logger_wrapper.get_logger()
    
    def run(self) -> DatasetDict:
        """
        Execute the complete pipeline
        
        Returns:
            Created DatasetDict
        """
        self.logger.info("=" * 80)
        self.logger.info("DATASET CREATION PIPELINE")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Load data
            input_config = self.config["input"]
            df = self.data_loader.load(
                file_path=Path(input_config["file_path"]),
                file_format=input_config["file_format"],
                **input_config.get("csv", {})
            )
            
            # Step 2: Process data
            processing_config = self.config["processing"]
            df = self.data_processor.process(
                df=df,
                anchor_col=input_config["columns"]["anchor"],
                positive_col=input_config["columns"]["positive"],
                cleaning_config=processing_config["cleaning"],
                validation_config=input_config["validation"]
            )
            
            # Step 3: Remove duplicates
            if processing_config["filtering"]["remove_duplicates"]:
                df, duplicate_count = self.data_processor.remove_duplicates(df)
            else:
                duplicate_count = 0
            
            # Step 4: Create splits
            splitting_config = processing_config["splitting"]
            splits = self.dataset_creator.create_splits(
                df=df,
                test_size=splitting_config["test_size"],
                validation_size=splitting_config["validation_size"],
                shuffle=splitting_config["shuffle"],
                seed=splitting_config["seed"]
            )
            
            # Step 5: Calculate statistics
            stats = self.stats_calculator.calculate(splits, duplicate_count)
            
            # Step 6: Display examples
            num_examples = self.config.get("display", {}).get("num_examples", 3)
            self.example_displayer.display_examples(splits, num_examples=num_examples)
            
            # Step 7: Create HuggingFace dataset
            dataset = self.dataset_creator.create_dataset_dict(splits)
            
            # Step 8: Save dataset
            output_config = self.config["output"]
            output_path = Path(output_config["output_dir"]) / output_config["dataset_name"]
            self.dataset_creator.save_dataset(
                dataset=dataset,
                output_dir=Path(output_config["output_dir"]),
                dataset_name=output_config["dataset_name"]
            )
            
            # Step 9: Save statistics
            if output_config.get("generate_statistics", True):
                stats_path = output_path / output_config.get("statistics_filename", "dataset_stats.json")
                stats.save(stats_path)
            
            # Step 10: Save example samples to file
            if self.config.get("display", {}).get("save_examples_to_file", True):
                num_file_examples = self.config.get("display", {}).get("num_file_examples", 5)
                self.example_displayer.save_examples_to_file(splits, output_path, num_examples=num_file_examples)
            
            self.logger.info("=" * 80)
            self.logger.info("DATASET CREATION COMPLETE")
            self.logger.info("=" * 80)
            
            return dataset
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("DATASET CREATION FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            raise


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point"""
    # Load configuration
    config = {
        "input": {
            "file_path": "data/raw/embedding_finetuning_dataset.csv",
            "file_format": "csv",
            "csv": {"delimiter": ",", "encoding": "utf-8"},
            "columns": {"anchor": "question", "positive": "context"},
            "validation": {
                "min_anchor_length": 1,
                "max_anchor_length": 512,
                "min_positive_length": 10,
                "max_positive_length": 2048
            }
        },
        "processing": {
            "cleaning": {
                "strip_whitespace": True,
                "normalize_spaces": True,
                "remove_empty_lines": True
            },
            "filtering": {"remove_duplicates": True},
            "splitting": {
                "test_size": 0.1,
                "validation_size": 0.0,
                "shuffle": True,
                "seed": 42
            }
        },
        "output": {
            "output_dir": "data/datasets",
            "dataset_name": "custom_embedding_dataset",
            "generate_statistics": True,
            "statistics_filename": "dataset_stats.json"
        },
        "display": {
            "num_examples": 3,
            "save_examples_to_file": True,
            "num_file_examples": 5
        },
        "logging": {
            "level": "INFO",
            "save_to_file": True,
            "log_dir": "data/logs"
        }
    }
    
    pipeline = DatasetCreationPipeline(config)
    dataset = pipeline.run()
    
    return dataset


if __name__ == "__main__":
    main()