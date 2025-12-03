"""
Production-Ready Fine-Tuning Script for Qwen3-Embedding-0.6B
Uses Unsloth for efficient QLoRA fine-tuning with Sentence Transformers
"""

import os
import json
import logging
import platform
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import torch
from datetime import datetime

# Import unsloth first to enable optimizations
import unsloth
from unsloth import FastModel

# Standard imports
from transformers import AutoModel
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import LoraConfig, TaskType

# Sentence Transformers imports
import sentence_transformers
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
    SentenceTransformer
)
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for the base model"""
    base_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    max_seq_length: int = 512
    load_in_4bit: bool = True
    trust_remote_code: bool = True  # Required for Qwen models
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")


@dataclass
class LoRAConfig:
    """Configuration for QLoRA adapters"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    use_rslora: bool = False
    target_modules: list = None
    exclude_modules: list = None
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    
    def __post_init__(self):
        """Set default target modules if not provided"""
        if self.target_modules is None:
            # Common attention modules - adapt based on model architecture
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.exclude_modules is None:
            self.exclude_modules = []
        
        # Validate
        if self.r <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Core training parameters
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 256
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    
    # Optimization
    optimizer: str = "adamw_torch_fused"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Evaluation and checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    logging_steps: int = 10
    
    # Output
    output_dir: str = f"data/fine_tuning/{datetime.now().strftime('%Y%m%d_%H%M%S')}/finetuned_qwen3_embedding"
    run_name: Optional[str] = None
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 0 if platform.system() == "Windows" else 4
    
    def __post_init__(self):
        """Set run name if not provided"""
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"qwen3_embedding_finetune_{timestamp}"


@dataclass
class DataConfig:
    """Configuration for dataset"""
    dataset_name: str = "philschmid/finanical-rag-embedding-dataset"
    dataset_split: str = "train"
    test_size: float = 0.1
    anchor_column: str = "question"
    positive_column: str = "context"
    
    def __post_init__(self):
        """Validate configuration"""
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Set up comprehensive logging
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("qwen3_finetuning")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with detailed formatting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"training_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Model Loading and Preparation
# ============================================================================

def load_base_model(
    config: ModelConfig,
    logger: logging.Logger
) -> Tuple[Any, Any]:
    """
    Load the base Qwen3 embedding model with Unsloth optimizations
    
    Args:
        config: Model configuration
        logger: Logger instance
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {config.base_model_id}")
    logger.info(f"Configuration: max_seq_length={config.max_seq_length}, "
                f"load_in_4bit={config.load_in_4bit}")
    
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.base_model_id,
            auto_model=AutoModel,  # Use AutoModel for automatic architecture detection
            max_seq_length=config.max_seq_length,
            dtype=None,  # Auto-detect optimal dtype (BF16/FP16)
            load_in_4bit=config.load_in_4bit,
            trust_remote_code=config.trust_remote_code,
        )
        
        logger.info(f"Successfully loaded {config.base_model_id}")
        logger.info(f"Model dtype: {model.dtype}")
        logger.info(f"Model device: {model.device}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def attach_lora_adapters(
    model: Any,
    config: LoRAConfig,
    logger: logging.Logger
) -> Any:
    """
    Attach QLoRA adapters to the base model
    
    Args:
        model: Base model
        config: LoRA configuration
        logger: Logger instance
    
    Returns:
        Model with LoRA adapters attached
    """
    logger.info("Attaching QLoRA adapters...")
    logger.info(f"LoRA config: r={config.r}, alpha={config.lora_alpha}, "
                f"dropout={config.lora_dropout}")
    logger.info(f"Target modules: {config.target_modules}")
    
    try:
        model = FastModel.get_peft_model(
            model,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            exclude_modules=config.exclude_modules,
            use_rslora=config.use_rslora,
            bias=config.bias,
            use_gradient_checkpointing="unsloth",
            modules_to_save=None,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable percentage: {trainable_percentage:.2f}%")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to attach LoRA adapters: {str(e)}")
        raise


def create_sentence_transformer_model(
    model: Any,
    tokenizer: Any,
    config: ModelConfig,
    logger: logging.Logger
) -> SentenceTransformer:
    """
    Wrap the Unsloth model in a SentenceTransformer for training
    
    Args:
        model: Unsloth model with LoRA adapters
        tokenizer: Model tokenizer
        config: Model configuration
        logger: Logger instance
    
    Returns:
        SentenceTransformer model ready for training
    """
    logger.info("Creating SentenceTransformer wrapper...")
    
    try:
        # 1. Create Transformer module
        transformer_module = sentence_transformers.models.Transformer(
            model_name_or_path=config.base_model_id,
            max_seq_length=config.max_seq_length,
        )
        
        # 2. Replace with our LoRA-patched model
        transformer_module.auto_model = model
        transformer_module.tokenizer = tokenizer
        
        logger.info("Assigned Unsloth LoRA model to Transformer module")
        
        # 3. Create Pooling module (mean pooling)
        hidden_size = model.config.hidden_size
        pooling_module = sentence_transformers.models.Pooling(
            word_embedding_dimension=hidden_size,
            pooling_mode="mean",
        )
        logger.info(f"Created Pooling module (mean pooling, hidden_size={hidden_size})")
        
        # 4. Create Normalize module
        normalize_module = sentence_transformers.models.Normalize()
        
        # 5. Assemble modules
        modules = [transformer_module, pooling_module, normalize_module]
        
        # 6. Create SentenceTransformer
        sbert_model = SentenceTransformer(modules=modules)
        
        logger.info("SentenceTransformer wrapper created successfully")
        
        return sbert_model
        
    except Exception as e:
        logger.error(f"Failed to create SentenceTransformer wrapper: {str(e)}")
        raise


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_and_prepare_dataset(
    config: DataConfig,
    logger: logging.Logger
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare the training dataset
    
    Args:
        config: Data configuration
        logger: Logger instance
    
    Returns:
        Tuple of (train_dataset, test_dataset, full_corpus_dataset)
    """
    logger.info(f"Loading dataset: {config.dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(
            config.dataset_name,
            split=config.dataset_split
        )
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Rename columns to match expected format
        dataset = dataset.rename_column(config.anchor_column, "anchor")
        dataset = dataset.rename_column(config.positive_column, "positive")
        
        # Add ID column
        dataset = dataset.add_column("id", range(len(dataset)))
        
        # Split into train and test
        split_dataset = dataset.train_test_split(
            test_size=config.test_size,
            seed=42
        )
        
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        
        logger.info(f"Train set: {len(train_dataset)} examples")
        logger.info(f"Test set: {len(test_dataset)} examples")
        
        # Create full corpus for evaluation
        corpus_dataset = concatenate_datasets([train_dataset, test_dataset])
        
        # Save test dataset for later evaluation
        return train_dataset, test_dataset, corpus_dataset
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


def create_evaluator(
    test_dataset: Dataset,
    corpus_dataset: Dataset,
    logger: logging.Logger
) -> InformationRetrievalEvaluator:
    """
    Create an evaluator for the embedding model
    
    Args:
        test_dataset: Test dataset
        corpus_dataset: Full corpus dataset
        logger: Logger instance
    
    Returns:
        Configured evaluator
    """
    logger.info("Creating Information Retrieval evaluator...")
    
    try:
        # Create corpus dictionary (id -> document)
        corpus = dict(
            zip(corpus_dataset["id"], corpus_dataset["positive"])
        )
        
        # Create queries dictionary (id -> question)
        queries = dict(
            zip(test_dataset["id"], test_dataset["anchor"])
        )
        
        # Create relevant documents mapping
        # Each query maps to its corresponding document (same ID)
        relevant_docs = {q_id: [q_id] for q_id in queries}
        
        # Create evaluator
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            score_functions={"cosine": cos_sim},
            name="ir-eval"
        )
        
        logger.info(f"Evaluator created with {len(queries)} queries and {len(corpus)} documents")
        
        return evaluator
        
    except Exception as e:
        logger.error(f"Failed to create evaluator: {str(e)}")
        raise


def save_evaluation_data(
    output_dir: Path,
    test_dataset: Dataset,
    corpus_dataset: Dataset,
    logger: logging.Logger
):
    """
    Save test and corpus data for later evaluation
    
    Args:
        output_dir: Output directory
        test_dataset: Test dataset
        corpus_dataset: Full corpus dataset
        logger: Logger instance
    """
    eval_data_dir = output_dir / "eval_data"
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    test_dataset.save_to_disk(str(eval_data_dir / "test_dataset"))
    corpus_dataset.save_to_disk(str(eval_data_dir / "corpus_dataset"))
    
    logger.info(f"Evaluation data saved to: {eval_data_dir}")


# ============================================================================
# Training Setup
# ============================================================================

def create_training_arguments(
    config: TrainingConfig,
    loss: Any,
    evaluator: Optional[InformationRetrievalEvaluator],
    logger: logging.Logger
) -> SentenceTransformerTrainingArguments:
    """
    Create training arguments for the trainer
    
    Args:
        config: Training configuration
        loss: Loss function being used
        evaluator: Optional evaluator
        logger: Logger instance
    
    Returns:
        Configured training arguments
    """
    logger.info("Creating training arguments...")
    
    # Determine batch sampler based on loss type
    batch_sampler = None
    if isinstance(loss, MultipleNegativesRankingLoss):
        batch_sampler = BatchSamplers.NO_DUPLICATES
        logger.info("Using NO_DUPLICATES batch sampler for MNRL loss")
    
    # Determine precision settings
    fp16 = not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_bf16_supported()
    logger.info(f"Precision: FP16={fp16}, BF16={bf16}")
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine metric for best model
    metric_for_best_model = None
    if evaluator and isinstance(evaluator, InformationRetrievalEvaluator):
        metric_for_best_model = "eval_ir-eval_cosine_ndcg@10"
    
    try:
        args = SentenceTransformerTrainingArguments(
            # Core training parameters
            output_dir=str(output_path),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            optim=config.optimizer,
            
            # Batch sampler
            batch_sampler=batch_sampler,
            
            # Precision
            fp16=fp16,
            bf16=bf16,
            tf32=True,
            fp16_full_eval=True,
            
            # Evaluation and saving
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True if evaluator else False,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=True,
            
            # Logging
            logging_steps=config.logging_steps,
            report_to=["tensorboard"],
            run_name=config.run_name,
            
            # Data loading
            dataloader_num_workers=config.dataloader_num_workers,
            dataloader_pin_memory=True,
            
            # Reproducibility
            seed=config.seed,
            
            # Performance
            auto_find_batch_size=False,  # Set to True if OOM issues
        )
        
        logger.info(f"Training arguments created successfully")
        logger.info(f"Total training steps: {config.num_train_epochs * config.gradient_accumulation_steps}")
        
        return args
        
    except Exception as e:
        logger.error(f"Failed to create training arguments: {str(e)}")
        raise


def save_configuration(
    output_dir: Path,
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    logger: logging.Logger
):
    """
    Save all configurations to JSON for reproducibility
    
    Args:
        output_dir: Output directory
        model_config: Model configuration
        lora_config: LoRA configuration
        training_config: Training configuration
        data_config: Data configuration
        logger: Logger instance
    """
    config_path = output_dir / "training_config.json"
    
    config_dict = {
        "model": asdict(model_config),
        "lora": asdict(lora_config),
        "training": asdict(training_config),
        "data": asdict(data_config),
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Initialize configurations
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig(
        num_train_epochs=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,
        weight_decay=0.1,
        eval_steps=4,
        save_steps=4,
        save_total_limit=2,
        logging_steps=2,
    )
    data_config = DataConfig()
    
    # Setup logging
    output_dir = Path(training_config.output_dir)
    log_dir = output_dir / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("QWEN3-EMBEDDING-0.6B FINE-TUNING WITH UNSLOTH")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Save configurations
        save_configuration(
            output_dir,
            model_config,
            lora_config,
            training_config,
            data_config,
            logger
        )
        
        # 1. Load base model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOADING BASE MODEL")
        logger.info("=" * 80)
        model, tokenizer = load_base_model(model_config, logger)
        
        # 2. Attach LoRA adapters
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: ATTACHING LORA ADAPTERS")
        logger.info("=" * 80)
        model = attach_lora_adapters(model, lora_config, logger)
        
        # 3. Create SentenceTransformer wrapper
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: CREATING SENTENCE TRANSFORMER WRAPPER")
        logger.info("=" * 80)
        sbert_model = create_sentence_transformer_model(
            model, tokenizer, model_config, logger
        )
        
        # 4. Load and prepare data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: LOADING AND PREPARING DATA")
        logger.info("=" * 80)
        train_dataset, test_dataset, corpus_dataset = load_and_prepare_dataset(
            data_config, logger
        )
        
        # Save evaluation data for later use
        save_evaluation_data(output_dir, test_dataset, corpus_dataset, logger)
        
        # 5. Create evaluator
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: CREATING EVALUATOR")
        logger.info("=" * 80)
        evaluator = create_evaluator(test_dataset, corpus_dataset, logger)
        
        # 6. Setup loss function
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: SETTING UP LOSS FUNCTION")
        logger.info("=" * 80)
        loss = MultipleNegativesRankingLoss(sbert_model)
        logger.info(f"Using loss: {type(loss).__name__}")
        logger.info("This loss automatically samples negatives from the same batch")
        
        # 7. Create training arguments
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: CREATING TRAINING ARGUMENTS")
        logger.info("=" * 80)
        args = create_training_arguments(
            training_config, loss, evaluator, logger
        )
        
        # 8. Initialize trainer
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: INITIALIZING TRAINER")
        logger.info("=" * 80)
        trainer = SentenceTransformerTrainer(
            model=sbert_model,
            args=args,
            train_dataset=train_dataset.select_columns(["anchor", "positive"]),
            eval_dataset=test_dataset.select_columns(["anchor", "positive"]),
            loss=loss,
            evaluator=evaluator,
        )
        logger.info("Trainer initialized successfully")
        
        # 9. Train the model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: TRAINING")
        logger.info("=" * 80)
        logger.info("Starting training...")
        
        train_result = trainer.train()
        
        logger.info("Training completed!")
        logger.info(f"Training time: {train_result.metrics.get('train_runtime', 0):.2f} seconds")
        
        # 10. Save the model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 10: SAVING MODEL")
        logger.info("=" * 80)
        sbert_model.save_pretrained(str(output_dir))
        logger.info(f"Model saved to: {output_dir}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Fine-tuned model saved to: {output_dir}")
        logger.info(f"Logs saved to: {log_dir}")
        logger.info(f"Training configuration saved to: {output_dir / 'training_config.json'}")
        logger.info(f"Evaluation data saved to: {output_dir / 'eval_data'}")
        logger.info("\nNext steps:")
        logger.info(f"  1. Run evaluation: python evaluate_qwen3_embedding.py --model-path {output_dir}")
        logger.info(f"  2. Compare with baseline: python evaluate_qwen3_embedding.py --model-path {output_dir} --compare")
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("TRAINING FAILED")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("\nGPU cache cleared")


if __name__ == "__main__":
    main()