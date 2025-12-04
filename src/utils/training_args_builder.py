import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Tuple, List
from transformers import EarlyStoppingCallback
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.config.config import ( 
    FT_TRAINING_NUM_TRAIN_EPOCHS,
    FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE,
    FT_TRAINING_PER_DEVICE_EVAL_BATCH_SIZE,
    FT_TRAINING_GRADIENT_ACCUMULATION_STEPS,
    FT_TRAINING_LEARNING_RATE,
    FT_TRAINING_LR_SCHEDULER_TYPE,
    FT_TRAINING_WARMUP_RATIO,
    FT_TRAINING_WARMUP_STEPS,
    FT_TRAINING_OPTIMIZER,
    FT_TRAINING_WEIGHT_DECAY,
    FT_TRAINING_ADAM_BETA1,
    FT_TRAINING_ADAM_BETA2,
    FT_TRAINING_ADAM_EPSILON,
    FT_TRAINING_MAX_GRAD_NORM,
    FT_TRAINING_EVAL_STRATEGY,
    FT_TRAINING_EVAL_STEPS,
    FT_TRAINING_EVAL_DELAY,
    FT_TRAINING_SAVE_STRATEGY,
    FT_TRAINING_SAVE_STEPS,
    FT_TRAINING_SAVE_TOTAL_LIMIT,
    FT_TRAINING_LOAD_BEST_MODEL_AT_END,
    FT_TRAINING_METRIC_FOR_BEST_MODEL,
    FT_TRAINING_GREATER_IS_BETTER,
    FT_TRAINING_LOGGING_STEPS,
    FT_TRAINING_LOGGING_FIRST_STEP,
    FT_TRAINING_LOGGING_STRATEGY,
    FT_TRAINING_RUN_NAME,
    FT_TRAINING_REPORT_TO,
    FT_TRAINING_SEED,
    FT_TRAINING_DATA_SEED,
    FT_TRAINING_DATALOADER_NUM_WORKERS,
    FT_TRAINING_DATALOADER_PIN_MEMORY,
    FT_TRAINING_DATALOADER_PREFETCH_FACTOR,
    FT_TRAINING_BATCH_SAMPLER,
    FT_TRAINING_FP16,
    FT_TRAINING_BF16,
    FT_TRAINING_TF32,
    FT_TRAINING_FP16_FULL_EVAL,
    FT_TRAINING_AUTO_FIND_BATCH_SIZE,
    FT_TRAINING_INCLUDE_INPUTS_FOR_METRICS,
    FT_TRAINING_LABEL_SMOOTHING_FACTOR,
    FT_TRAINING_PREDICTION_LOSS_ONLY,
    FT_EVALUATION_IR_NAME,
    FT_MONITORING_EARLY_STOPPING_ENABLED,
    FT_MONITORING_EARLY_STOPPING_PATIENCE,
    FT_MONITORING_EARLY_STOPPING_MIN_DELTA,
)


class TrainingArgumentsBuilder:
    """
    Builds training arguments from configuration
    Can be moved to core/training_args_builder.py
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize training arguments builder"""
        self.logger = logger or logging.getLogger(__name__)
        self.early_stopping_callback = None
    
    def build(
        self,
        output_dir: Path,
        loss: Any,
        evaluator: Optional[InformationRetrievalEvaluator]
    ) -> Tuple[SentenceTransformerTrainingArguments, List]:
        """
        Build training arguments and callbacks
        
        Args:
            output_dir: Output directory
            loss: Loss function
            evaluator: Optional evaluator
        
        Returns:
            Tuple of (training arguments, list of callbacks)
        """
        self.logger.info("=" * 80)
        self.logger.info("CREATING TRAINING ARGUMENTS")
        self.logger.info("=" * 80)
        
        # Determine batch sampler
        batch_sampler = None
        if FT_TRAINING_BATCH_SAMPLER == "NO_DUPLICATES":
            batch_sampler = BatchSamplers.NO_DUPLICATES
        elif FT_TRAINING_BATCH_SAMPLER == "GROUP_BY_LABEL":
            batch_sampler = BatchSamplers.GROUP_BY_LABEL
        
        if batch_sampler:
            self.logger.info(f"Batch sampler: {FT_TRAINING_BATCH_SAMPLER}")
        
        # Determine precision settings
        fp16 = FT_TRAINING_FP16
        bf16 = FT_TRAINING_BF16
        
        if fp16 is None and bf16 is None:
            fp16 = not torch.cuda.is_bf16_supported()
            bf16 = torch.cuda.is_bf16_supported()
        
        self.logger.info(f"Precision: FP16={fp16}, BF16={bf16}, TF32={FT_TRAINING_TF32}")
        
        # Determine metric for best model
        metric_for_best_model = FT_TRAINING_METRIC_FOR_BEST_MODEL
        if evaluator and not metric_for_best_model:
            metric_for_best_model = f"eval_{FT_EVALUATION_IR_NAME}_cosine_ndcg@10"
        
        # Determine run name
        run_name = FT_TRAINING_RUN_NAME
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"qwen3_embedding_finetune_{timestamp}"
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping callback
        if FT_MONITORING_EARLY_STOPPING_ENABLED and evaluator:
            self.early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=FT_MONITORING_EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=FT_MONITORING_EARLY_STOPPING_MIN_DELTA
            )
            callbacks.append(self.early_stopping_callback)
            self.logger.info(f"Early stopping enabled: patience={FT_MONITORING_EARLY_STOPPING_PATIENCE}, "
                           f"threshold={FT_MONITORING_EARLY_STOPPING_MIN_DELTA}")
        
        try:
            args = SentenceTransformerTrainingArguments(
                # Core training parameters
                output_dir=str(output_dir),
                num_train_epochs=FT_TRAINING_NUM_TRAIN_EPOCHS,
                per_device_train_batch_size=FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=FT_TRAINING_PER_DEVICE_EVAL_BATCH_SIZE,
                gradient_accumulation_steps=FT_TRAINING_GRADIENT_ACCUMULATION_STEPS,
                
                # Learning rate
                learning_rate=FT_TRAINING_LEARNING_RATE,
                lr_scheduler_type=FT_TRAINING_LR_SCHEDULER_TYPE,
                warmup_ratio=FT_TRAINING_WARMUP_RATIO,
                warmup_steps=FT_TRAINING_WARMUP_STEPS if FT_TRAINING_WARMUP_STEPS is not None else 0,
                
                # Optimizer
                optim=FT_TRAINING_OPTIMIZER,
                weight_decay=FT_TRAINING_WEIGHT_DECAY,
                adam_beta1=FT_TRAINING_ADAM_BETA1,
                adam_beta2=FT_TRAINING_ADAM_BETA2,
                adam_epsilon=FT_TRAINING_ADAM_EPSILON,
                max_grad_norm=FT_TRAINING_MAX_GRAD_NORM,
                
                # Batch sampler
                batch_sampler=batch_sampler,
                
                # Precision
                fp16=fp16,
                bf16=bf16,
                tf32=FT_TRAINING_TF32,
                fp16_full_eval=FT_TRAINING_FP16_FULL_EVAL,
                
                # Evaluation and saving
                eval_strategy=FT_TRAINING_EVAL_STRATEGY,
                eval_steps=FT_TRAINING_EVAL_STEPS,
                eval_delay=FT_TRAINING_EVAL_DELAY,
                save_strategy=FT_TRAINING_SAVE_STRATEGY,
                save_steps=FT_TRAINING_SAVE_STEPS,
                save_total_limit=FT_TRAINING_SAVE_TOTAL_LIMIT,
                load_best_model_at_end=FT_TRAINING_LOAD_BEST_MODEL_AT_END if evaluator else False,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=FT_TRAINING_GREATER_IS_BETTER,
                
                # Logging
                logging_steps=FT_TRAINING_LOGGING_STEPS,
                logging_first_step=FT_TRAINING_LOGGING_FIRST_STEP,
                logging_strategy=FT_TRAINING_LOGGING_STRATEGY,
                report_to=FT_TRAINING_REPORT_TO,
                run_name=run_name,
                
                # Data loading
                dataloader_num_workers=FT_TRAINING_DATALOADER_NUM_WORKERS,
                dataloader_pin_memory=FT_TRAINING_DATALOADER_PIN_MEMORY,
                dataloader_prefetch_factor=FT_TRAINING_DATALOADER_PREFETCH_FACTOR if FT_TRAINING_DATALOADER_NUM_WORKERS > 0 else None,
                
                # Reproducibility
                seed=FT_TRAINING_SEED,
                data_seed=FT_TRAINING_DATA_SEED or FT_TRAINING_SEED,
                
                # Performance
                auto_find_batch_size=FT_TRAINING_AUTO_FIND_BATCH_SIZE,
                
                # Advanced
                include_inputs_for_metrics=FT_TRAINING_INCLUDE_INPUTS_FOR_METRICS,
                label_smoothing_factor=FT_TRAINING_LABEL_SMOOTHING_FACTOR,
                prediction_loss_only=FT_TRAINING_PREDICTION_LOSS_ONLY,
            )
            
            self.logger.info("[OK] Training arguments created")
            self.logger.info(f"  Run name: {run_name}")
            self.logger.info(f"  Output dir: {output_dir}")
            self.logger.info(f"  Epochs: {FT_TRAINING_NUM_TRAIN_EPOCHS}")
            self.logger.info(f"  Train batch size: {FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE}")
            self.logger.info(f"  Learning rate: {FT_TRAINING_LEARNING_RATE}")
            self.logger.info(f"  Callbacks: {len(callbacks)}")
            
            return args, callbacks
            
        except Exception as e:
            self.logger.error(f"Failed to create training arguments: {str(e)}", exc_info=True)
            raise