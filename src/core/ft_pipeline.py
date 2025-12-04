"""
Production-Grade Fine-Tuning Script for Qwen3-Embedding Models with Unsloth

This module provides a robust fine-tuning framework with:
- Configuration-driven architecture
- Class-based design for reusability
- Comprehensive monitoring and logging
- Memory-efficient QLoRA fine-tuning
- Evaluation during and after training
- Modular components ready for extraction to utils
"""
import warnings
warnings.filterwarnings('ignore', message='Unexpected key')

from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.utils.logging import FinetuningLogger
from src.utils.model_loader import ModelLoader
from src.utils.loss_factory import LossFactory
from src.utils.data_models import TrainingMetrics
from src.utils. hardware_utils import HardwareManager
from src.utils.dataset_manager import DatasetManager
from src.utils.config_manager import ConfigurationManager
from src.utils.evaluator_factory import EvaluatorFactory
from src.utils.training_args_builder import TrainingArgumentsBuilder
from src.config.config import (
    FT_MODEL_BASE_MODEL_ID,
    FT_LORA_R,
    FT_TRAINING_NUM_TRAIN_EPOCHS,
    FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE,
    FT_TRAINING_LEARNING_RATE,
    FT_TRAINING_OUTPUT_DIR,
    FT_DATASET_DATASET_NAME,
    FT_EVALUATION_IR_SCORE_FUNCTION,
    FT_EVALUATION_IR_NAME,
    FT_EVALUATION_SAVE_EVAL_DATA,
    FT_OUTPUT_CREATE_SUBDIRS,
    FT_OUTPUT_SUBDIRS,
    FT_OUTPUT_LOGGING_SAVE_TO_FILE,
    FT_OUTPUT_LOGGING_LOG_FILENAME,
    FT_OUTPUT_SAVE_METRICS,
    FT_OUTPUT_METRICS_FILENAME,
    FT_OUTPUT_SAVE_MODEL_INFO,
    FT_OUTPUT_MODEL_INFO_FILENAME,
    FT_POST_TRAINING_SAVE_FINAL_MODEL,
    FT_POST_TRAINING_UPLOAD_TO_HUB,
    FT_POST_TRAINING_HUB_REPO_ID,
)



class FineTuningPipeline:
    """
    Main pipeline orchestrating the entire fine-tuning process
    """
    
    def __init__(self):
        """Initialize fine-tuning pipeline"""
        self._setup_output_directory()
        self._setup_logging()
        
        # Initialize components
        self.hardware_manager = HardwareManager(logger=self.logger)
        self.model_loader = ModelLoader(logger=self.logger)
        self.dataset_manager = DatasetManager(logger=self.logger)
        self.evaluator_factory = EvaluatorFactory
        self.loss_factory = LossFactory(logger=self.logger)
        self.training_args_builder = TrainingArgumentsBuilder(logger=self.logger)
        self.config_manager = ConfigurationManager(logger=self.logger)
    
    def _setup_output_directory(self) -> None:
        """Setup output directory structure"""
        # Replace {timestamp} in output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_str = FT_TRAINING_OUTPUT_DIR.replace("{timestamp}", timestamp)
        self.output_dir = Path(output_dir_str)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        if FT_OUTPUT_CREATE_SUBDIRS:
            for subdir in FT_OUTPUT_SUBDIRS:
                (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = None
        if FT_OUTPUT_LOGGING_SAVE_TO_FILE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = FT_OUTPUT_LOGGING_LOG_FILENAME.replace("{timestamp}", timestamp)
            log_file = self.output_dir / "logs" / log_filename
        
        logger_wrapper = FinetuningLogger(log_file=log_file)
        self.logger = logger_wrapper.get_logger()
        
        self._log_configuration()
    
    def _log_configuration(self) -> None:
        """Log pipeline configuration"""
        self.logger.info("=" * 80)
        self.logger.info("QWEN3-EMBEDDING FINE-TUNING PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("\nConfiguration:")
        self.logger.info(f"  Base model: {FT_MODEL_BASE_MODEL_ID}")
        self.logger.info(f"  LoRA rank: {FT_LORA_R}")
        self.logger.info(f"  Epochs: {FT_TRAINING_NUM_TRAIN_EPOCHS}")
        self.logger.info(f"  Batch size: {FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE}")
        self.logger.info(f"  Learning rate: {FT_TRAINING_LEARNING_RATE}")
        self.logger.info(f"  Dataset: {FT_DATASET_DATASET_NAME}")
    
    def run(self) -> TrainingMetrics:
        """
        Execute the complete fine-tuning pipeline
        
        Returns:
            TrainingMetrics with training results
        """
        try:
            # Log hardware info
            self.hardware_manager.log_hardware_info()
            
            # Save configuration
            self.config_manager.save_config(self.output_dir)
            
            # Step 1: Load base model
            model, tokenizer = self.model_loader.load_base_model()
            
            # Step 2: Attach LoRA adapters
            model = self.model_loader.attach_lora_adapters(model)
            
            # Step 3: Create SentenceTransformer
            sbert_model = self.model_loader.create_sentence_transformer(model, tokenizer)
            
            # Step 4: Load dataset
            train_dataset, test_dataset, corpus_dataset = self.dataset_manager.load_dataset()
            
            # Save evaluation data
            self.dataset_manager.save_evaluation_data(
                self.output_dir, test_dataset, corpus_dataset
            )
            
            # Step 5: Create evaluator
            self.logger.info("\n" + "=" * 80)
            self.logger.info("CREATING EVALUATOR")
            self.logger.info("=" * 80)
            # Use the static method that accepts datasets directly
            # Create corpus and queries from datasets
            corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
            queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
            relevant_docs = {q_id: [q_id] for q_id in queries}

            from sentence_transformers.util import cos_sim
            from sentence_transformers.evaluation import InformationRetrievalEvaluator

            score_functions = {"cosine": cos_sim} if FT_EVALUATION_IR_SCORE_FUNCTION == "cosine" else {}

            evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                score_functions=score_functions,
                name=FT_EVALUATION_IR_NAME
            )

            self.logger.info(f"[OK] Evaluator created: {len(queries)} queries, {len(corpus)} documents")
            self.logger.info(f"  Queries: {len(queries)}")
            self.logger.info(f"  Corpus: {len(corpus)}")

            # Step 6: Create loss function
            loss = self.loss_factory.create_loss(sbert_model)
            
            # Step 7: Create training arguments
            args, callbacks = self.training_args_builder.build(self.output_dir, loss, evaluator)
            
            # Step 8: Initialize trainer
            self.logger.info("\n" + "=" * 80)
            self.logger.info("INITIALIZING TRAINER")
            self.logger.info("=" * 80)
            
            trainer = SentenceTransformerTrainer(
                model=sbert_model,
                args=args,
                train_dataset=train_dataset.select_columns(["anchor", "positive"]),
                eval_dataset=test_dataset.select_columns(["anchor", "positive"]),
                loss=loss,
                evaluator=evaluator,
                callbacks=callbacks,
            )
            self.logger.info("[OK] Trainer initialized")
            
            # Step 9: Train
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STARTING TRAINING")
            self.logger.info("=" * 80)
            
            train_result = trainer.train()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("TRAINING COMPLETE")
            self.logger.info("=" * 80)
            
            # Extract metrics
            metrics = TrainingMetrics(
                train_runtime=train_result.metrics.get('train_runtime', 0),
                train_samples_per_second=train_result.metrics.get('train_samples_per_second', 0),
                train_steps_per_second=train_result.metrics.get('train_steps_per_second', 0),
                total_flos=train_result.metrics.get('total_flos', 0),
                train_loss=train_result.metrics.get('train_loss', 0),
                epoch=train_result.metrics.get('epoch', 0),
            )
            
            self.logger.info(f"Training time: {metrics.train_runtime:.2f} seconds")
            self.logger.info(f"Samples/second: {metrics.train_samples_per_second:.2f}")
            self.logger.info(f"Final loss: {metrics.train_loss:.4f}")
            
            # Step 10: Save model
            if FT_POST_TRAINING_SAVE_FINAL_MODEL:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("SAVING FINAL MODEL")
                self.logger.info("=" * 80)
                sbert_model.save_pretrained(str(self.output_dir))
                self.logger.info(f"[OK] Model saved to: {self.output_dir}")
            
            # Save metrics
            if FT_OUTPUT_SAVE_METRICS:
                metrics.save(self.output_dir / FT_OUTPUT_METRICS_FILENAME)
                self.logger.info(f"[OK] Metrics saved")
            
            # Save model info
            if FT_OUTPUT_SAVE_MODEL_INFO and hasattr(self.model_loader, 'model_info'):
                self.model_loader.model_info.save(self.output_dir / FT_OUTPUT_MODEL_INFO_FILENAME)
                self.logger.info(f"[OK] Model info saved")
            
            # Final summary
            self._log_summary()
            
            return metrics
            
        except Exception as e:
            self.logger.error("\n" + "=" * 80)
            self.logger.error("TRAINING FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {str(e)}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            self.hardware_manager.clear_cache()
            self.logger.info("\n[OK] GPU cache cleared")
    
    def _log_summary(self) -> None:
        """Log training summary"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Model saved: {FT_POST_TRAINING_SAVE_FINAL_MODEL}")
        self.logger.info(f"Evaluation data saved: {FT_EVALUATION_SAVE_EVAL_DATA}")
        self.logger.info("\nNext steps:")
        self.logger.info(f"  1. Evaluate model: python scripts/evaluate.py")
        self.logger.info(f"  2. Compare with baseline")
        if FT_POST_TRAINING_UPLOAD_TO_HUB and FT_POST_TRAINING_HUB_REPO_ID:
            self.logger.info(f"  3. Upload to Hub: python scripts/push_to_hub.py")
