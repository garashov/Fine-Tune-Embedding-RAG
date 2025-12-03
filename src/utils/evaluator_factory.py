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
        
        # Load datasets
        test_dataset = load_from_disk(str(eval_data_dir / "test_dataset"))
        corpus_dataset = load_from_disk(str(eval_data_dir / "corpus_dataset"))
        
        if logger:
            logger.info(f"Loaded {len(test_dataset)} test examples")
            logger.info(f"Loaded {len(corpus_dataset)} corpus documents")
        
        # Create corpus dictionary
        corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
        
        # Create queries dictionary
        queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
        
        # Create relevant documents mapping
        relevant_docs = {q_id: [q_id] for q_id in queries}
        
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
