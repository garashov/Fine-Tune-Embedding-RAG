import logging
from pathlib import Path
from typing import Optional
from datasets import load_from_disk, load_dataset
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.config.config import (
    EVAL_DATA_HF_DATASET_NAME,
    EVAL_DATA_HF_DATASET_SPLIT,
    EVAL_DATA_HF_TEST_SIZE,
    EVAL_DATA_HF_ANCHOR_COLUMN,
    EVAL_DATA_HF_POSITIVE_COLUMN,
    EVAL_DATA_HF_MAX_SAMPLES,
    EVAL_DATA_HF_SEED,
    EVAL_METRICS_SCORE_FUNCTION,
    EVAL_METRICS_EVALUATOR_NAME,
)



class EvaluatorFactory:
    """
    Factory class for creating InformationRetrievalEvaluator instances
    Can be moved to utils/evaluator_factory.py
    """
    
    @staticmethod
    def from_local_data(
        eval_data_dir: Path,
        score_function: str = EVAL_METRICS_SCORE_FUNCTION,
        evaluator_name: str = EVAL_METRICS_EVALUATOR_NAME,
        logger: Optional[logging.Logger] = None
    ) -> InformationRetrievalEvaluator:
        """
        Create evaluator from saved local evaluation data
        
        Args:
            eval_data_dir: Directory containing saved evaluation datasets
            score_function: Scoring function to use
            evaluator_name: Name for the evaluator
            logger: Optional logger instance
        
        Returns:
            Configured InformationRetrievalEvaluator
        
        Raises:
            FileNotFoundError: If evaluation data not found
            ValueError: If data is invalid
        """
        if logger:
            logger.info(f"Loading evaluation data from: {eval_data_dir}")
        
        if not eval_data_dir.exists():
            raise FileNotFoundError(f"Evaluation data directory not found: {eval_data_dir}")
        
        # Load the test dataset (which contains both anchor and positive)
        try:
            # Try loading as a DatasetDict first
            from datasets import load_from_disk
            dataset = load_from_disk(str(eval_data_dir))
            
            # Use test split if available, otherwise use train split
            if "test" in dataset:
                test_dataset = dataset["test"]
            elif "train" in dataset:
                test_dataset = dataset["train"]
                if logger:
                    logger.warning("No test split found, using train split for evaluation")
            else:
                raise ValueError("Dataset must contain either 'test' or 'train' split")
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to load dataset: {str(e)}")
            raise
        
        if logger:
            logger.info(f"Loaded {len(test_dataset)} test examples")
        
        # Validate required columns
        required_cols = {"id", "anchor", "positive"}
        missing_cols = required_cols - set(test_dataset.column_names)
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")
        
        # Create corpus dictionary (id -> positive text)
        corpus = {}
        for example in test_dataset:
            doc_id = str(example["id"])
            corpus[doc_id] = example["positive"]
        
        # Create queries dictionary (id -> anchor text)
        queries = {}
        for example in test_dataset:
            query_id = str(example["id"])
            queries[query_id] = example["anchor"]
        
        # Create relevant documents mapping (query_id -> [doc_id])
        # Since each query corresponds to exactly one document with the same ID
        relevant_docs = {query_id: [query_id] for query_id in queries}
        
        if logger:
            logger.info(f"Created corpus with {len(corpus)} documents")
            logger.info(f"Created queries with {len(queries)} queries")
        
        # Configure score function
        score_functions = {"cosine": cos_sim} if score_function == "cosine" else {}
        
        # Create evaluator
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            score_functions=score_functions,
            name=evaluator_name
        )
        
        if logger:
            logger.info(f"Evaluator created: {len(queries)} queries, {len(corpus)} documents")
        
        return evaluator
    
    @staticmethod
    def from_huggingface_dataset(
        dataset_name: str = EVAL_DATA_HF_DATASET_NAME,
        dataset_split: str = EVAL_DATA_HF_DATASET_SPLIT,
        test_size: float = EVAL_DATA_HF_TEST_SIZE,
        anchor_column: str = EVAL_DATA_HF_ANCHOR_COLUMN,
        positive_column: str = EVAL_DATA_HF_POSITIVE_COLUMN,
        max_samples: Optional[int] = EVAL_DATA_HF_MAX_SAMPLES,
        seed: int = EVAL_DATA_HF_SEED,
        score_function: str = EVAL_METRICS_SCORE_FUNCTION,
        evaluator_name: str = EVAL_METRICS_EVALUATOR_NAME,
        logger: Optional[logging.Logger] = None
    ) -> InformationRetrievalEvaluator:
        """
        Create evaluator from HuggingFace dataset
        
        Args:
            dataset_name: HuggingFace dataset identifier
            dataset_split: Dataset split to use
            test_size: Proportion for test split
            anchor_column: Column name for queries
            positive_column: Column name for documents
            max_samples: Optional limit on samples
            seed: Random seed for reproducibility
            score_function: Scoring function to use
            evaluator_name: Name for the evaluator
            logger: Optional logger instance
        
        Returns:
            Configured InformationRetrievalEvaluator
        """
        if logger:
            logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=dataset_split)
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=seed).select(range(max_samples))
            if logger:
                logger.info(f"Limited dataset to {max_samples} samples")
        
        # Split into train/test
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        test_dataset = dataset['test']
        
        if logger:
            logger.info(f"Using {len(test_dataset)} test examples")
        
        # Create corpus, queries, and relevant docs
        corpus = {}
        queries = {}
        relevant_docs = {}
        
        for idx, example in enumerate(test_dataset):
            doc_id = f"doc_{idx}"
            query_id = f"query_{idx}"
            
            corpus[doc_id] = example[positive_column]
            queries[query_id] = example[anchor_column]
            relevant_docs[query_id] = [doc_id]
        
        # Configure score function
        score_functions = {"cosine": cos_sim} if score_function == "cosine" else {}
        
        # Create evaluator
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            score_functions=score_functions,
            name=evaluator_name
        )
        
        if logger:
            logger.info(f"Evaluator created: {len(queries)} queries, {len(corpus)} documents")
        
        return evaluator
