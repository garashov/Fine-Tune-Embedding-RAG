import os
import json
from src.config import (
    EVAL_CONFIG, 
    FT_CONFIG, 
    HF_PUSH_CONFIG
)

# --------------------------------------------------------------------------------------------
#                                    Evaluation Config
# --------------------------------------------------------------------------------------------
# =====================================
# MODEL CONFIGURATION
# =====================================
EVAL_MODEL_FT_PATH = EVAL_CONFIG["model.fine_tuned_path"]
EVAL_MODEL_BASELINE_ID = EVAL_CONFIG["model.baseline_model"]

# =====================================
# EVALUATION OPTIONS
# =====================================
EVAL_EVALUATE_FINETUNED = EVAL_CONFIG["evaluation.evaluate_finetuned"]
EVAL_EVALUATE_BASELINE = EVAL_CONFIG["evaluation.evaluate_baseline"]
EVAL_COMPARE_MODELS = EVAL_CONFIG["evaluation.compare_models"]
EVAL_DEVICE = EVAL_CONFIG["evaluation.device"]

# =====================================
# DATA SOURCE CONFIGURATION
# =====================================
EVAL_DATA_USE_LOCAL_EVAL_DATA = EVAL_CONFIG["data_source.use_local_eval_data"]
EVAL_DATA_USE_HUGGINGFACE_DATASET = EVAL_CONFIG["data_source.use_huggingface_dataset"]

# Local evaluation data settings
EVAL_DATA_LOCAL_EVAL_DATA_DIR = EVAL_CONFIG["data_source.local_eval_data.eval_data_dir"]

# HuggingFace dataset settings
EVAL_DATA_HF_DATASET_NAME = EVAL_CONFIG["data_source.huggingface_dataset.dataset_name"]
EVAL_DATA_HF_DATASET_SPLIT = EVAL_CONFIG["data_source.huggingface_dataset.dataset_split"]
EVAL_DATA_HF_TEST_SIZE = EVAL_CONFIG["data_source.huggingface_dataset.test_size"]
EVAL_DATA_HF_ANCHOR_COLUMN = EVAL_CONFIG["data_source.huggingface_dataset.anchor_column"]
EVAL_DATA_HF_POSITIVE_COLUMN = EVAL_CONFIG["data_source.huggingface_dataset.positive_column"]
EVAL_DATA_HF_MAX_SAMPLES = EVAL_CONFIG["data_source.huggingface_dataset.max_samples"]
EVAL_DATA_HF_SEED = EVAL_CONFIG["data_source.huggingface_dataset.seed"]

# =====================================
# EVALUATION METRICS CONFIGURATION
# =====================================
EVAL_METRICS_SCORE_FUNCTION = EVAL_CONFIG["metrics.score_function"]
EVAL_METRICS_KEY_METRICS = EVAL_CONFIG["metrics.key_metrics"]
EVAL_METRICS_EVALUATOR_NAME = EVAL_CONFIG["metrics.evaluator_name"]

# =====================================
# OUTPUT CONFIGURATION
# =====================================
EVAL_OUTPUT_DIR = EVAL_CONFIG["output.output_dir"]

# Logging configuration
EVAL_OUTPUT_LOGGING_LEVEL = EVAL_CONFIG["output.logging.level"]
EVAL_OUTPUT_LOGGING_SAVE_TO_FILE = EVAL_CONFIG["output.logging.save_to_file"]
EVAL_OUTPUT_LOGGING_LOG_SUBDIR = EVAL_CONFIG["output.logging.log_subdir"]
EVAL_OUTPUT_LOGGING_CONSOLE_FORMAT = EVAL_CONFIG["output.logging.console_format"]
EVAL_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT = EVAL_CONFIG["output.logging.console_date_format"]
EVAL_OUTPUT_LOGGING_FILE_FORMAT = EVAL_CONFIG["output.logging.file_format"]

# Result file naming
EVAL_OUTPUT_RESULTS_INCLUDE_TIMESTAMP = EVAL_CONFIG["output.results.include_timestamp"]
EVAL_OUTPUT_RESULTS_BASELINE_RESULTS = EVAL_CONFIG["output.results.baseline_results"]
EVAL_OUTPUT_RESULTS_FINETUNED_RESULTS = EVAL_CONFIG["output.results.finetuned_results"]
EVAL_OUTPUT_RESULTS_COMPARISON_RESULTS = EVAL_CONFIG["output.results.comparison_results"]

# =====================================
# PERFORMANCE CONFIGURATION
# =====================================
EVAL_PERF_CLEAR_CUDA_CACHE = EVAL_CONFIG["performance.clear_cuda_cache"]
EVAL_PERF_FORCE_GARBAGE_COLLECTION = EVAL_CONFIG["performance.force_garbage_collection"]
EVAL_PERF_REPORT_GPU_MEMORY = EVAL_CONFIG["performance.report_gpu_memory"]

# =====================================
# ADVANCED OPTIONS
# =====================================
EVAL_ADVANCED_USE_INFERENCE_MODE = EVAL_CONFIG["advanced.use_inference_mode"]
EVAL_ADVANCED_EVAL_MODE = EVAL_CONFIG["advanced.eval_mode"]



# --------------------------------------------------------------------------------------------
#                              HuggingFace Hub Push Config
# --------------------------------------------------------------------------------------------
# =====================================
# MODEL CONFIGURATION
# =====================================
HF_PUSH_MODEL_FT_PATH = HF_PUSH_CONFIG["model.fine_tuned_path"]

# =====================================
# HUGGINGFACE HUB CONFIGURATION
# =====================================
HF_PUSH_REPO_ID = HF_PUSH_CONFIG["huggingface_hub.repo_id"]
HF_PUSH_PRIVATE = HF_PUSH_CONFIG["huggingface_hub.private"]
HF_PUSH_COMMIT_MESSAGE = HF_PUSH_CONFIG["huggingface_hub.commit_message"]
HF_PUSH_CREATE_PR = HF_PUSH_CONFIG["huggingface_hub.create_pr"]
HF_PUSH_REVISION = HF_PUSH_CONFIG["huggingface_hub.revision"]

# Repository metadata
HF_PUSH_METADATA_DESCRIPTION = HF_PUSH_CONFIG["huggingface_hub.metadata.description"]
HF_PUSH_METADATA_TAGS = HF_PUSH_CONFIG["huggingface_hub.metadata.tags"]
HF_PUSH_METADATA_LANGUAGE = HF_PUSH_CONFIG["huggingface_hub.metadata.language"]
HF_PUSH_METADATA_LICENSE = HF_PUSH_CONFIG["huggingface_hub.metadata.license"]
HF_PUSH_METADATA_TASKS = HF_PUSH_CONFIG["huggingface_hub.metadata.tasks"]
HF_PUSH_METADATA_DATASETS = HF_PUSH_CONFIG["huggingface_hub.metadata.datasets"]
HF_PUSH_METADATA_BASE_MODEL = HF_PUSH_CONFIG["huggingface_hub.metadata.base_model"]

# =====================================
# AUTHENTICATION CONFIGURATION
# =====================================
HF_PUSH_AUTH_USE_AUTH_TOKEN = HF_PUSH_CONFIG["authentication.use_auth_token"]
HF_PUSH_AUTH_TOKEN_SOURCE = HF_PUSH_CONFIG["authentication.token_source"]
HF_PUSH_AUTH_TOKEN_ENV_VAR = HF_PUSH_CONFIG["authentication.token_env_var"]

# =====================================
# UPLOAD CONFIGURATION
# =====================================
HF_PUSH_UPLOAD_MAX_RETRIES = HF_PUSH_CONFIG["upload.max_retries"]
HF_PUSH_UPLOAD_RETRY_DELAY_SECONDS = HF_PUSH_CONFIG["upload.retry_delay_seconds"]
HF_PUSH_UPLOAD_TIMEOUT = HF_PUSH_CONFIG["upload.upload_timeout"]
HF_PUSH_UPLOAD_SHOW_PROGRESS = HF_PUSH_CONFIG["upload.show_progress"]
HF_PUSH_UPLOAD_EXCLUDE_FILES = HF_PUSH_CONFIG["upload.exclude_files"]
HF_PUSH_UPLOAD_USE_LFS = HF_PUSH_CONFIG["upload.use_lfs"]
HF_PUSH_UPLOAD_LFS_THRESHOLD_MB = HF_PUSH_CONFIG["upload.lfs_threshold_mb"]

# =====================================
# MODEL CARD CONFIGURATION
# =====================================
HF_PUSH_MODEL_CARD_GENERATE_CARD = HF_PUSH_CONFIG["model_card.generate_card"]
HF_PUSH_MODEL_CARD_TEMPLATE = HF_PUSH_CONFIG["model_card.template"]
HF_PUSH_MODEL_CARD_INCLUDE_SECTIONS = HF_PUSH_CONFIG["model_card.include_sections"]
HF_PUSH_MODEL_CARD_CUSTOM_CONTENT = HF_PUSH_CONFIG["model_card.custom_content"]

# =====================================
# OUTPUT CONFIGURATION
# =====================================
# Logging
HF_PUSH_OUTPUT_LOGGING_LEVEL = HF_PUSH_CONFIG["output.logging.level"]
HF_PUSH_OUTPUT_LOGGING_SAVE_TO_FILE = HF_PUSH_CONFIG["output.logging.save_to_file"]
HF_PUSH_OUTPUT_LOGGING_LOG_DIR = HF_PUSH_CONFIG["output.logging.log_dir"]
HF_PUSH_OUTPUT_LOGGING_LOG_FILENAME = HF_PUSH_CONFIG["output.logging.log_filename"]
HF_PUSH_OUTPUT_LOGGING_CONSOLE_FORMAT = HF_PUSH_CONFIG["output.logging.console_format"]
HF_PUSH_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT = HF_PUSH_CONFIG["output.logging.console_date_format"]
HF_PUSH_OUTPUT_LOGGING_FILE_FORMAT = HF_PUSH_CONFIG["output.logging.file_format"]

# Notifications
HF_PUSH_OUTPUT_NOTIFICATIONS_ENABLED = HF_PUSH_CONFIG["output.notifications.enabled"]
HF_PUSH_OUTPUT_NOTIFICATIONS_METHODS = HF_PUSH_CONFIG["output.notifications.methods"]
HF_PUSH_OUTPUT_NOTIFICATIONS_EMAIL = HF_PUSH_CONFIG["output.notifications.email"]
HF_PUSH_OUTPUT_NOTIFICATIONS_SLACK = HF_PUSH_CONFIG["output.notifications.slack"]
HF_PUSH_OUTPUT_NOTIFICATIONS_WEBHOOK = HF_PUSH_CONFIG["output.notifications.webhook"]

# =====================================
# PRE-PUSH VALIDATION
# =====================================
HF_PUSH_VALIDATION_ENABLED = HF_PUSH_CONFIG["validation.enabled"]
HF_PUSH_VALIDATION_CHECKS = HF_PUSH_CONFIG["validation.checks"]
HF_PUSH_VALIDATION_STRICT_MODE = HF_PUSH_CONFIG["validation.strict_mode"]
HF_PUSH_VALIDATION_TIMEOUT = HF_PUSH_CONFIG["validation.timeout"]

# =====================================
# BACKUP CONFIGURATION
# =====================================
HF_PUSH_BACKUP_CREATE_BACKUP = HF_PUSH_CONFIG["backup.create_backup"]
HF_PUSH_BACKUP_BACKUP_DIR = HF_PUSH_CONFIG["backup.backup_dir"]
HF_PUSH_BACKUP_BACKUP_NAME = HF_PUSH_CONFIG["backup.backup_name"]
HF_PUSH_BACKUP_MAX_BACKUPS = HF_PUSH_CONFIG["backup.max_backups"]

# =====================================
# POST-PUSH ACTIONS
# =====================================
HF_PUSH_POST_PUSH_ACTIONS = HF_PUSH_CONFIG["post_push.actions"]
HF_PUSH_POST_PUSH_VERIFY_UPLOAD_ENABLED = HF_PUSH_CONFIG["post_push.verify_upload.enabled"]
HF_PUSH_POST_PUSH_VERIFY_UPLOAD_TEST_DOWNLOAD = HF_PUSH_CONFIG["post_push.verify_upload.test_download"]
HF_PUSH_POST_PUSH_VERIFY_UPLOAD_VERIFY_CHECKSUMS = HF_PUSH_CONFIG["post_push.verify_upload.verify_checksums"]

# =====================================
# ADVANCED OPTIONS
# =====================================
HF_PUSH_ADVANCED_USE_SAFETENSORS = HF_PUSH_CONFIG["advanced.use_safetensors"]
HF_PUSH_ADVANCED_COMMIT_STRATEGY = HF_PUSH_CONFIG["advanced.commit_strategy"]
HF_PUSH_ADVANCED_PARALLEL_UPLOADS = HF_PUSH_CONFIG["advanced.parallel_uploads"]
HF_PUSH_ADVANCED_MAX_WORKERS = HF_PUSH_CONFIG["advanced.max_workers"]
HF_PUSH_ADVANCED_RESUME_ON_FAILURE = HF_PUSH_CONFIG["advanced.resume_on_failure"]
HF_PUSH_ADVANCED_CACHE_DIR = HF_PUSH_CONFIG["advanced.cache_dir"]
HF_PUSH_ADVANCED_FORCE_PUSH = HF_PUSH_CONFIG["advanced.force_push"]


# --------------------------------------------------------------------------------------------
#                              Fine-Tuning Configuration
# --------------------------------------------------------------------------------------------
# =====================================
# MODEL CONFIGURATION
# =====================================
FT_MODEL_BASE_MODEL_ID = FT_CONFIG["model.base_model_id"]
FT_MODEL_MAX_SEQ_LENGTH = FT_CONFIG["model.max_seq_length"]
FT_MODEL_LOAD_IN_4BIT = FT_CONFIG["model.load_in_4bit"]
FT_MODEL_TRUST_REMOTE_CODE = FT_CONFIG["model.trust_remote_code"]
FT_MODEL_DTYPE = FT_CONFIG["model.dtype"]
FT_MODEL_DEVICE_MAP = FT_CONFIG["model.device_map"]

# =====================================
# LORA/QLORA CONFIGURATION
# =====================================
FT_LORA_R = FT_CONFIG["lora.r"]
FT_LORA_ALPHA = FT_CONFIG["lora.lora_alpha"]
FT_LORA_DROPOUT = FT_CONFIG["lora.lora_dropout"]
FT_LORA_USE_RSLORA = FT_CONFIG["lora.use_rslora"]
FT_LORA_TARGET_MODULES = FT_CONFIG["lora.target_modules"]
FT_LORA_EXCLUDE_MODULES = FT_CONFIG["lora.exclude_modules"]
FT_LORA_BIAS = FT_CONFIG["lora.bias"]
FT_LORA_TASK_TYPE = FT_CONFIG["lora.task_type"]
FT_LORA_USE_GRADIENT_CHECKPOINTING = FT_CONFIG["lora.use_gradient_checkpointing"]
FT_LORA_MODULES_TO_SAVE = FT_CONFIG["lora.modules_to_save"]

# =====================================
# TRAINING CONFIGURATION
# =====================================
# Core training parameters
FT_TRAINING_NUM_TRAIN_EPOCHS = FT_CONFIG["training.num_train_epochs"]
FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE = FT_CONFIG["training.per_device_train_batch_size"]
FT_TRAINING_PER_DEVICE_EVAL_BATCH_SIZE = FT_CONFIG["training.per_device_eval_batch_size"]
FT_TRAINING_GRADIENT_ACCUMULATION_STEPS = FT_CONFIG["training.gradient_accumulation_steps"]

# Learning rate settings
FT_TRAINING_LEARNING_RATE = FT_CONFIG["training.learning_rate"]
FT_TRAINING_LR_SCHEDULER_TYPE = FT_CONFIG["training.lr_scheduler_type"]
FT_TRAINING_WARMUP_RATIO = FT_CONFIG["training.warmup_ratio"]
FT_TRAINING_WARMUP_STEPS = FT_CONFIG["training.warmup_steps"]

# Optimizer settings
FT_TRAINING_OPTIMIZER = FT_CONFIG["training.optimizer"]
FT_TRAINING_WEIGHT_DECAY = FT_CONFIG["training.weight_decay"]
FT_TRAINING_ADAM_BETA1 = FT_CONFIG["training.adam_beta1"]
FT_TRAINING_ADAM_BETA2 = FT_CONFIG["training.adam_beta2"]
FT_TRAINING_ADAM_EPSILON = FT_CONFIG["training.adam_epsilon"]
FT_TRAINING_MAX_GRAD_NORM = FT_CONFIG["training.max_grad_norm"]

# Evaluation and checkpointing
FT_TRAINING_EVAL_STRATEGY = FT_CONFIG["training.eval_strategy"]
FT_TRAINING_EVAL_STEPS = FT_CONFIG["training.eval_steps"]
FT_TRAINING_EVAL_DELAY = FT_CONFIG["training.eval_delay"]
FT_TRAINING_SAVE_STRATEGY = FT_CONFIG["training.save_strategy"]
FT_TRAINING_SAVE_STEPS = FT_CONFIG["training.save_steps"]
FT_TRAINING_SAVE_TOTAL_LIMIT = FT_CONFIG["training.save_total_limit"]

# Best model tracking
FT_TRAINING_LOAD_BEST_MODEL_AT_END = FT_CONFIG["training.load_best_model_at_end"]
FT_TRAINING_METRIC_FOR_BEST_MODEL = FT_CONFIG["training.metric_for_best_model"]
FT_TRAINING_GREATER_IS_BETTER = FT_CONFIG["training.greater_is_better"]

# Logging
FT_TRAINING_LOGGING_STEPS = FT_CONFIG["training.logging_steps"]
FT_TRAINING_LOGGING_FIRST_STEP = FT_CONFIG["training.logging_first_step"]
FT_TRAINING_LOGGING_STRATEGY = FT_CONFIG["training.logging_strategy"]

# Output and reporting
FT_TRAINING_OUTPUT_DIR = FT_CONFIG["training.output_dir"]
FT_TRAINING_RUN_NAME = FT_CONFIG["training.run_name"]
FT_TRAINING_REPORT_TO = FT_CONFIG["training.report_to"]

# Reproducibility
FT_TRAINING_SEED = FT_CONFIG["training.seed"]
FT_TRAINING_DATA_SEED = FT_CONFIG["training.data_seed"]

# Performance optimization
FT_TRAINING_DATALOADER_NUM_WORKERS = FT_CONFIG["training.dataloader_num_workers"]
FT_TRAINING_DATALOADER_PIN_MEMORY = FT_CONFIG["training.dataloader_pin_memory"]
FT_TRAINING_DATALOADER_PREFETCH_FACTOR = FT_CONFIG["training.dataloader_prefetch_factor"]

# Batch sampler
FT_TRAINING_BATCH_SAMPLER = FT_CONFIG["training.batch_sampler"]

# Mixed precision training
FT_TRAINING_FP16 = FT_CONFIG["training.fp16"]
FT_TRAINING_BF16 = FT_CONFIG["training.bf16"]
FT_TRAINING_TF32 = FT_CONFIG["training.tf32"]
FT_TRAINING_FP16_FULL_EVAL = FT_CONFIG["training.fp16_full_eval"]

# Memory optimization
FT_TRAINING_AUTO_FIND_BATCH_SIZE = FT_CONFIG["training.auto_find_batch_size"]
FT_TRAINING_GRADIENT_CHECKPOINTING = FT_CONFIG["training.gradient_checkpointing"]

# Advanced options
FT_TRAINING_INCLUDE_INPUTS_FOR_METRICS = FT_CONFIG["training.include_inputs_for_metrics"]
FT_TRAINING_LABEL_SMOOTHING_FACTOR = FT_CONFIG["training.label_smoothing_factor"]
FT_TRAINING_PREDICTION_LOSS_ONLY = FT_CONFIG["training.prediction_loss_only"]

# =====================================
# DATASET CONFIGURATION
# =====================================
FT_DATASET_DATASET_NAME = FT_CONFIG["dataset.dataset_name"]
FT_DATASET_DATASET_SPLIT = FT_CONFIG["dataset.dataset_split"]
FT_DATASET_TEST_SIZE = FT_CONFIG["dataset.test_size"]
FT_DATASET_ANCHOR_COLUMN = FT_CONFIG["dataset.anchor_column"]
FT_DATASET_POSITIVE_COLUMN = FT_CONFIG["dataset.positive_column"]

# Data preprocessing
FT_DATASET_SHUFFLE_TRAIN = FT_CONFIG["dataset.shuffle_train"]
FT_DATASET_SHUFFLE_SEED = FT_CONFIG["dataset.shuffle_seed"]

# Optional data limits
FT_DATASET_MAX_TRAIN_SAMPLES = FT_CONFIG["dataset.max_train_samples"]
FT_DATASET_MAX_EVAL_SAMPLES = FT_CONFIG["dataset.max_eval_samples"]

# Caching
FT_DATASET_CACHE_DIR = FT_CONFIG["dataset.cache_dir"]
FT_DATASET_KEEP_IN_MEMORY = FT_CONFIG["dataset.keep_in_memory"]

# =====================================
# LOSS FUNCTION CONFIGURATION
# =====================================
FT_LOSS_TYPE = FT_CONFIG["loss.type"]

# MultipleNegativesRankingLoss settings
FT_LOSS_MNRL_SCALE = FT_CONFIG["loss.mnrl.scale"]
FT_LOSS_MNRL_SIMILARITY_FCT = FT_CONFIG["loss.mnrl.similarity_fct"]

# CosineSimilarityLoss settings
FT_LOSS_COSINE_LOSS_FCT = FT_CONFIG["loss.cosine.loss_fct"]

# =====================================
# EVALUATION CONFIGURATION
# =====================================
FT_EVALUATION_EVALUATOR_TYPE = FT_CONFIG["evaluation.evaluator_type"]

# IR Evaluator settings
FT_EVALUATION_IR_SCORE_FUNCTION = FT_CONFIG["evaluation.ir_evaluator.score_function"]
FT_EVALUATION_IR_METRICS = FT_CONFIG["evaluation.ir_evaluator.metrics"]
FT_EVALUATION_IR_NAME = FT_CONFIG["evaluation.ir_evaluator.name"]
FT_EVALUATION_IR_SAVE_CORPUS = FT_CONFIG["evaluation.ir_evaluator.save_corpus"]
FT_EVALUATION_IR_SAVE_QUERIES = FT_CONFIG["evaluation.ir_evaluator.save_queries"]

# Save evaluation data
FT_EVALUATION_SAVE_EVAL_DATA = FT_CONFIG["evaluation.save_eval_data"]
FT_EVALUATION_EVAL_DATA_SUBDIR = FT_CONFIG["evaluation.eval_data_subdir"]

# =====================================
# OUTPUT CONFIGURATION
# =====================================
FT_OUTPUT_CREATE_SUBDIRS = FT_CONFIG["output.create_subdirs"]
FT_OUTPUT_SUBDIRS = FT_CONFIG["output.subdirs"]

# Logging
FT_OUTPUT_LOGGING_LEVEL = FT_CONFIG["output.logging.level"]
FT_OUTPUT_LOGGING_SAVE_TO_FILE = FT_CONFIG["output.logging.save_to_file"]
FT_OUTPUT_LOGGING_LOG_FILENAME = FT_CONFIG["output.logging.log_filename"]
FT_OUTPUT_LOGGING_CONSOLE_FORMAT = FT_CONFIG["output.logging.console_format"]
FT_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT = FT_CONFIG["output.logging.console_date_format"]
FT_OUTPUT_LOGGING_FILE_FORMAT = FT_CONFIG["output.logging.file_format"]

# Save configuration
FT_OUTPUT_SAVE_CONFIG = FT_CONFIG["output.save_config"]
FT_OUTPUT_CONFIG_FILENAME = FT_CONFIG["output.config_filename"]

# Save training metrics
FT_OUTPUT_SAVE_METRICS = FT_CONFIG["output.save_metrics"]
FT_OUTPUT_METRICS_FILENAME = FT_CONFIG["output.metrics_filename"]

# Save model information
FT_OUTPUT_SAVE_MODEL_INFO = FT_CONFIG["output.save_model_info"]
FT_OUTPUT_MODEL_INFO_FILENAME = FT_CONFIG["output.model_info_filename"]

# =====================================
# HARDWARE CONFIGURATION
# =====================================
FT_HARDWARE_CUDA_VISIBLE_DEVICES = FT_CONFIG["hardware.cuda_visible_devices"]
FT_HARDWARE_USE_MULTI_GPU = FT_CONFIG["hardware.use_multi_gpu"]
FT_HARDWARE_LOCAL_RANK = FT_CONFIG["hardware.local_rank"]
FT_HARDWARE_EMPTY_CUDA_CACHE_STEPS = FT_CONFIG["hardware.empty_cuda_cache_steps"]
FT_HARDWARE_MAX_MEMORY = FT_CONFIG["hardware.max_memory"]
FT_HARDWARE_NUM_THREADS = FT_CONFIG["hardware.num_threads"]

# =====================================
# MONITORING AND DEBUGGING
# =====================================
FT_MONITORING_SHOW_PROGRESS_BAR = FT_CONFIG["monitoring.show_progress_bar"]
FT_MONITORING_LOG_MEMORY_USAGE = FT_CONFIG["monitoring.log_memory_usage"]
FT_MONITORING_MEMORY_LOG_INTERVAL = FT_CONFIG["monitoring.memory_log_interval"]
FT_MONITORING_LOG_MODEL_STATS = FT_CONFIG["monitoring.log_model_stats"]
FT_MONITORING_MODEL_STATS_INTERVAL = FT_CONFIG["monitoring.model_stats_interval"]
FT_MONITORING_LOG_GRADIENT_NORM = FT_CONFIG["monitoring.log_gradient_norm"]

# Early stopping
FT_MONITORING_EARLY_STOPPING_ENABLED = FT_CONFIG["monitoring.early_stopping.enabled"]
FT_MONITORING_EARLY_STOPPING_PATIENCE = FT_CONFIG["monitoring.early_stopping.patience"]
FT_MONITORING_EARLY_STOPPING_MIN_DELTA = FT_CONFIG["monitoring.early_stopping.min_delta"]

# =====================================
# POST-TRAINING ACTIONS
# =====================================
FT_POST_TRAINING_SAVE_FINAL_MODEL = FT_CONFIG["post_training.save_final_model"]
FT_POST_TRAINING_SAVE_MERGED_MODEL = FT_CONFIG["post_training.save_merged_model"]

# Model upload
FT_POST_TRAINING_UPLOAD_TO_HUB = FT_CONFIG["post_training.upload_to_hub"]
FT_POST_TRAINING_HUB_REPO_ID = FT_CONFIG["post_training.hub_repo_id"]
FT_POST_TRAINING_HUB_PRIVATE = FT_CONFIG["post_training.hub_private"]

# Evaluation
FT_POST_TRAINING_RUN_FINAL_EVALUATION = FT_CONFIG["post_training.run_final_evaluation"]
FT_POST_TRAINING_COMPARE_WITH_BASELINE = FT_CONFIG["post_training.compare_with_baseline"]
FT_POST_TRAINING_BASELINE_MODEL = FT_CONFIG["post_training.baseline_model"]

# Cleanup
FT_POST_TRAINING_DELETE_CHECKPOINTS = FT_CONFIG["post_training.delete_checkpoints"]
FT_POST_TRAINING_KEEP_BEST_CHECKPOINT = FT_CONFIG["post_training.keep_best_checkpoint"]

# =====================================
# ADVANCED OPTIONS
# =====================================
# Unsloth-specific
FT_ADVANCED_UNSLOTH_USE_UNSLOTH = FT_CONFIG["advanced.unsloth.use_unsloth"]
FT_ADVANCED_UNSLOTH_MAX_SEQ_LENGTH_OPTIMIZATION = FT_CONFIG["advanced.unsloth.max_seq_length_optimization"]

# Distributed training
FT_ADVANCED_DISTRIBUTED_ENABLED = FT_CONFIG["advanced.distributed.enabled"]
FT_ADVANCED_DISTRIBUTED_BACKEND = FT_CONFIG["advanced.distributed.backend"]
FT_ADVANCED_DISTRIBUTED_FIND_UNUSED_PARAMETERS = FT_CONFIG["advanced.distributed.find_unused_parameters"]

# Compilation
FT_ADVANCED_TORCH_COMPILE = FT_CONFIG["advanced.torch_compile"]
FT_ADVANCED_TORCH_COMPILE_BACKEND = FT_CONFIG["advanced.torch_compile_backend"]
FT_ADVANCED_TORCH_COMPILE_MODE = FT_CONFIG["advanced.torch_compile_mode"]

# Profiling
FT_ADVANCED_PROFILING_ENABLED = FT_CONFIG["advanced.profiling.enabled"]
FT_ADVANCED_PROFILING_PROFILE_MEMORY = FT_CONFIG["advanced.profiling.profile_memory"]
FT_ADVANCED_PROFILING_PROFILE_STEPS = FT_CONFIG["advanced.profiling.profile_steps"]
FT_ADVANCED_PROFILING_OUTPUT_DIR = FT_CONFIG["advanced.profiling.output_dir"]

# Debug options
FT_ADVANCED_DEBUG_ENABLED = FT_CONFIG["advanced.debug.enabled"]
FT_ADVANCED_DEBUG_DEBUG_OVERFLOW = FT_CONFIG["advanced.debug.debug_overflow"]
FT_ADVANCED_DEBUG_DEBUG_UNDERFLOW = FT_CONFIG["advanced.debug.debug_underflow"]




# =====================================
# OPTIONAL: Print all variables for verification
# =====================================
if __name__ == "__main__":
    print("=" * 80)
    print("EVAL CONFIGURATION")
    print("=" * 80)
    print(f"MODEL_FT_PATH: {EVAL_MODEL_FT_PATH}")
    print(f"MODEL_BASELINE_ID: {EVAL_MODEL_BASELINE_ID}")
    
    print("\n" + "=" * 80)
    print("EVALUATION OPTIONS")
    print("=" * 80)
    print(f"EVAL_EVALUATE_FINETUNED: {EVAL_EVALUATE_FINETUNED}")
    print(f"EVAL_EVALUATE_BASELINE: {EVAL_EVALUATE_BASELINE}")
    print(f"EVAL_COMPARE_MODELS: {EVAL_COMPARE_MODELS}")
    print(f"EVAL_DEVICE: {EVAL_DEVICE}")
    
    print("\n" + "=" * 80)
    print("DATA SOURCE CONFIGURATION")
    print("=" * 80)
    print(f"DATA_USE_LOCAL_EVAL_DATA: {EVAL_DATA_USE_LOCAL_EVAL_DATA}")
    print(f"DATA_USE_HUGGINGFACE_DATASET: {EVAL_DATA_USE_HUGGINGFACE_DATASET}")
    print(f"DATA_LOCAL_EVAL_DATA_DIR: {EVAL_DATA_LOCAL_EVAL_DATA_DIR}")
    print(f"DATA_HF_DATASET_NAME: {EVAL_DATA_HF_DATASET_NAME}")
    print(f"DATA_HF_DATASET_SPLIT: {EVAL_DATA_HF_DATASET_SPLIT}")
    print(f"DATA_HF_TEST_SIZE: {EVAL_DATA_HF_TEST_SIZE}")
    print(f"DATA_HF_ANCHOR_COLUMN: {EVAL_DATA_HF_ANCHOR_COLUMN}")
    print(f"DATA_HF_POSITIVE_COLUMN: {EVAL_DATA_HF_POSITIVE_COLUMN}")
    print(f"DATA_HF_MAX_SAMPLES: {EVAL_DATA_HF_MAX_SAMPLES}")
    print(f"DATA_HF_SEED: {EVAL_DATA_HF_SEED}")
    
    print("\n" + "=" * 80)
    print("EVALUATION METRICS CONFIGURATION")
    print("=" * 80)
    print(f"METRICS_SCORE_FUNCTION: {EVAL_METRICS_SCORE_FUNCTION}")
    print(f"METRICS_KEY_METRICS: {EVAL_METRICS_KEY_METRICS}")
    print(f"METRICS_EVALUATOR_NAME: {EVAL_METRICS_EVALUATOR_NAME}")
    
    print("\n" + "=" * 80)
    print("OUTPUT CONFIGURATION")
    print("=" * 80)
    print(f"OUTPUT_DIR: {EVAL_OUTPUT_DIR}")
    print(f"OUTPUT_LOGGING_LEVEL: {EVAL_OUTPUT_LOGGING_LEVEL}")
    print(f"OUTPUT_LOGGING_SAVE_TO_FILE: {EVAL_OUTPUT_LOGGING_SAVE_TO_FILE}")
    print(f"OUTPUT_LOGGING_LOG_SUBDIR: {EVAL_OUTPUT_LOGGING_LOG_SUBDIR}")
    print(f"OUTPUT_LOGGING_CONSOLE_FORMAT: {EVAL_OUTPUT_LOGGING_CONSOLE_FORMAT}")
    print(f"OUTPUT_LOGGING_CONSOLE_DATE_FORMAT: {EVAL_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT}")
    print(f"OUTPUT_LOGGING_FILE_FORMAT: {EVAL_OUTPUT_LOGGING_FILE_FORMAT}")
    print(f"OUTPUT_RESULTS_INCLUDE_TIMESTAMP: {EVAL_OUTPUT_RESULTS_INCLUDE_TIMESTAMP}")
    print(f"OUTPUT_RESULTS_BASELINE_RESULTS: {EVAL_OUTPUT_RESULTS_BASELINE_RESULTS}")
    print(f"OUTPUT_RESULTS_FINETUNED_RESULTS: {EVAL_OUTPUT_RESULTS_FINETUNED_RESULTS}")
    print(f"OUTPUT_RESULTS_COMPARISON_RESULTS: {EVAL_OUTPUT_RESULTS_COMPARISON_RESULTS}")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE CONFIGURATION")
    print("=" * 80)
    print(f"PERF_CLEAR_CUDA_CACHE: {EVAL_PERF_CLEAR_CUDA_CACHE}")
    print(f"PERF_FORCE_GARBAGE_COLLECTION: {EVAL_PERF_FORCE_GARBAGE_COLLECTION}")
    print(f"PERF_REPORT_GPU_MEMORY: {EVAL_PERF_REPORT_GPU_MEMORY}")
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIONS")
    print("=" * 80)
    print(f"ADVANCED_USE_INFERENCE_MODE: {EVAL_ADVANCED_USE_INFERENCE_MODE}")
    print(f"ADVANCED_EVAL_MODE: {EVAL_ADVANCED_EVAL_MODE}")
    
    
    print("=" * 80)
    print("HF PUSH CONFIGURATION")
    print("=" * 80)
    print(f"MODEL_FT_PATH: {HF_PUSH_MODEL_FT_PATH}")
    
    print("\n" + "=" * 80)
    print("HUGGINGFACE HUB CONFIGURATION")
    print("=" * 80)
    print(f"REPO_ID: {HF_PUSH_REPO_ID}")
    print(f"PRIVATE: {HF_PUSH_PRIVATE}")
    print(f"COMMIT_MESSAGE: {HF_PUSH_COMMIT_MESSAGE}")
    print(f"CREATE_PR: {HF_PUSH_CREATE_PR}")
    print(f"REVISION: {HF_PUSH_REVISION}")
    
    print("\n" + "=" * 80)
    print("REPOSITORY METADATA")
    print("=" * 80)
    print(f"DESCRIPTION: {HF_PUSH_METADATA_DESCRIPTION}")
    print(f"TAGS: {HF_PUSH_METADATA_TAGS}")
    print(f"LANGUAGE: {HF_PUSH_METADATA_LANGUAGE}")
    print(f"LICENSE: {HF_PUSH_METADATA_LICENSE}")
    print(f"TASKS: {HF_PUSH_METADATA_TASKS}")
    print(f"DATASETS: {HF_PUSH_METADATA_DATASETS}")
    print(f"BASE_MODEL: {HF_PUSH_METADATA_BASE_MODEL}")
    
    print("\n" + "=" * 80)
    print("AUTHENTICATION CONFIGURATION")
    print("=" * 80)
    print(f"USE_AUTH_TOKEN: {HF_PUSH_AUTH_USE_AUTH_TOKEN}")
    print(f"TOKEN_SOURCE: {HF_PUSH_AUTH_TOKEN_SOURCE}")
    print(f"TOKEN_ENV_VAR: {HF_PUSH_AUTH_TOKEN_ENV_VAR}")
    
    print("\n" + "=" * 80)
    print("UPLOAD CONFIGURATION")
    print("=" * 80)
    print(f"MAX_RETRIES: {HF_PUSH_UPLOAD_MAX_RETRIES}")
    print(f"RETRY_DELAY_SECONDS: {HF_PUSH_UPLOAD_RETRY_DELAY_SECONDS}")
    print(f"TIMEOUT: {HF_PUSH_UPLOAD_TIMEOUT}")
    print(f"SHOW_PROGRESS: {HF_PUSH_UPLOAD_SHOW_PROGRESS}")
    print(f"EXCLUDE_FILES: {HF_PUSH_UPLOAD_EXCLUDE_FILES}")
    print(f"USE_LFS: {HF_PUSH_UPLOAD_USE_LFS}")
    print(f"LFS_THRESHOLD_MB: {HF_PUSH_UPLOAD_LFS_THRESHOLD_MB}")
    
    print("\n" + "=" * 80)
    print("MODEL CARD CONFIGURATION")
    print("=" * 80)
    print(f"GENERATE_CARD: {HF_PUSH_MODEL_CARD_GENERATE_CARD}")
    print(f"TEMPLATE: {HF_PUSH_MODEL_CARD_TEMPLATE}")
    print(f"INCLUDE_SECTIONS: {HF_PUSH_MODEL_CARD_INCLUDE_SECTIONS}")
    
    print("\n" + "=" * 80)
    print("OUTPUT CONFIGURATION - LOGGING")
    print("=" * 80)
    print(f"LEVEL: {HF_PUSH_OUTPUT_LOGGING_LEVEL}")
    print(f"SAVE_TO_FILE: {HF_PUSH_OUTPUT_LOGGING_SAVE_TO_FILE}")
    print(f"LOG_DIR: {HF_PUSH_OUTPUT_LOGGING_LOG_DIR}")
    print(f"LOG_FILENAME: {HF_PUSH_OUTPUT_LOGGING_LOG_FILENAME}")
    
    print("\n" + "=" * 80)
    print("OUTPUT CONFIGURATION - NOTIFICATIONS")
    print("=" * 80)
    print(f"ENABLED: {HF_PUSH_OUTPUT_NOTIFICATIONS_ENABLED}")
    print(f"METHODS: {HF_PUSH_OUTPUT_NOTIFICATIONS_METHODS}")
    
    print("\n" + "=" * 80)
    print("PRE-PUSH VALIDATION")
    print("=" * 80)
    print(f"ENABLED: {HF_PUSH_VALIDATION_ENABLED}")
    print(f"CHECKS: {HF_PUSH_VALIDATION_CHECKS}")
    print(f"STRICT_MODE: {HF_PUSH_VALIDATION_STRICT_MODE}")
    print(f"TIMEOUT: {HF_PUSH_VALIDATION_TIMEOUT}")
    
    print("\n" + "=" * 80)
    print("BACKUP CONFIGURATION")
    print("=" * 80)
    print(f"CREATE_BACKUP: {HF_PUSH_BACKUP_CREATE_BACKUP}")
    print(f"BACKUP_DIR: {HF_PUSH_BACKUP_BACKUP_DIR}")
    print(f"BACKUP_NAME: {HF_PUSH_BACKUP_BACKUP_NAME}")
    print(f"MAX_BACKUPS: {HF_PUSH_BACKUP_MAX_BACKUPS}")
    
    print("\n" + "=" * 80)
    print("POST-PUSH ACTIONS")
    print("=" * 80)
    print(f"ACTIONS: {HF_PUSH_POST_PUSH_ACTIONS}")
    print(f"VERIFY_UPLOAD_ENABLED: {HF_PUSH_POST_PUSH_VERIFY_UPLOAD_ENABLED}")
    print(f"TEST_DOWNLOAD: {HF_PUSH_POST_PUSH_VERIFY_UPLOAD_TEST_DOWNLOAD}")
    print(f"VERIFY_CHECKSUMS: {HF_PUSH_POST_PUSH_VERIFY_UPLOAD_VERIFY_CHECKSUMS}")
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIONS")
    print("=" * 80)
    print(f"USE_SAFETENSORS: {HF_PUSH_ADVANCED_USE_SAFETENSORS}")
    print(f"COMMIT_STRATEGY: {HF_PUSH_ADVANCED_COMMIT_STRATEGY}")
    print(f"PARALLEL_UPLOADS: {HF_PUSH_ADVANCED_PARALLEL_UPLOADS}")
    print(f"MAX_WORKERS: {HF_PUSH_ADVANCED_MAX_WORKERS}")
    print(f"RESUME_ON_FAILURE: {HF_PUSH_ADVANCED_RESUME_ON_FAILURE}")
    print(f"CACHE_DIR: {HF_PUSH_ADVANCED_CACHE_DIR}")
    print(f"FORCE_PUSH: {HF_PUSH_ADVANCED_FORCE_PUSH}")
    
    print("=" * 80)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 80)
    print(f"BASE_MODEL_ID: {FT_MODEL_BASE_MODEL_ID}")
    print(f"MAX_SEQ_LENGTH: {FT_MODEL_MAX_SEQ_LENGTH}")
    print(f"LOAD_IN_4BIT: {FT_MODEL_LOAD_IN_4BIT}")
    print(f"TRUST_REMOTE_CODE: {FT_MODEL_TRUST_REMOTE_CODE}")
    print(f"DTYPE: {FT_MODEL_DTYPE}")
    print(f"DEVICE_MAP: {FT_MODEL_DEVICE_MAP}")
    
    print("\n" + "=" * 80)
    print("LORA/QLORA CONFIGURATION")
    print("=" * 80)
    print(f"R: {FT_LORA_R}")
    print(f"ALPHA: {FT_LORA_ALPHA}")
    print(f"DROPOUT: {FT_LORA_DROPOUT}")
    print(f"USE_RSLORA: {FT_LORA_USE_RSLORA}")
    print(f"TARGET_MODULES: {FT_LORA_TARGET_MODULES}")
    print(f"EXCLUDE_MODULES: {FT_LORA_EXCLUDE_MODULES}")
    print(f"BIAS: {FT_LORA_BIAS}")
    print(f"TASK_TYPE: {FT_LORA_TASK_TYPE}")
    print(f"USE_GRADIENT_CHECKPOINTING: {FT_LORA_USE_GRADIENT_CHECKPOINTING}")
    print(f"MODULES_TO_SAVE: {FT_LORA_MODULES_TO_SAVE}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - CORE")
    print("=" * 80)
    print(f"NUM_TRAIN_EPOCHS: {FT_TRAINING_NUM_TRAIN_EPOCHS}")
    print(f"PER_DEVICE_TRAIN_BATCH_SIZE: {FT_TRAINING_PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"PER_DEVICE_EVAL_BATCH_SIZE: {FT_TRAINING_PER_DEVICE_EVAL_BATCH_SIZE}")
    print(f"GRADIENT_ACCUMULATION_STEPS: {FT_TRAINING_GRADIENT_ACCUMULATION_STEPS}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - LEARNING RATE")
    print("=" * 80)
    print(f"LEARNING_RATE: {FT_TRAINING_LEARNING_RATE}")
    print(f"LR_SCHEDULER_TYPE: {FT_TRAINING_LR_SCHEDULER_TYPE}")
    print(f"WARMUP_RATIO: {FT_TRAINING_WARMUP_RATIO}")
    print(f"WARMUP_STEPS: {FT_TRAINING_WARMUP_STEPS}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - OPTIMIZER")
    print("=" * 80)
    print(f"OPTIMIZER: {FT_TRAINING_OPTIMIZER}")
    print(f"WEIGHT_DECAY: {FT_TRAINING_WEIGHT_DECAY}")
    print(f"ADAM_BETA1: {FT_TRAINING_ADAM_BETA1}")
    print(f"ADAM_BETA2: {FT_TRAINING_ADAM_BETA2}")
    print(f"ADAM_EPSILON: {FT_TRAINING_ADAM_EPSILON}")
    print(f"MAX_GRAD_NORM: {FT_TRAINING_MAX_GRAD_NORM}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - EVALUATION & CHECKPOINTING")
    print("=" * 80)
    print(f"EVAL_STRATEGY: {FT_TRAINING_EVAL_STRATEGY}")
    print(f"EVAL_STEPS: {FT_TRAINING_EVAL_STEPS}")
    print(f"EVAL_DELAY: {FT_TRAINING_EVAL_DELAY}")
    print(f"SAVE_STRATEGY: {FT_TRAINING_SAVE_STRATEGY}")
    print(f"SAVE_STEPS: {FT_TRAINING_SAVE_STEPS}")
    print(f"SAVE_TOTAL_LIMIT: {FT_TRAINING_SAVE_TOTAL_LIMIT}")
    print(f"LOAD_BEST_MODEL_AT_END: {FT_TRAINING_LOAD_BEST_MODEL_AT_END}")
    print(f"METRIC_FOR_BEST_MODEL: {FT_TRAINING_METRIC_FOR_BEST_MODEL}")
    print(f"GREATER_IS_BETTER: {FT_TRAINING_GREATER_IS_BETTER}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - LOGGING")
    print("=" * 80)
    print(f"LOGGING_STEPS: {FT_TRAINING_LOGGING_STEPS}")
    print(f"LOGGING_FIRST_STEP: {FT_TRAINING_LOGGING_FIRST_STEP}")
    print(f"LOGGING_STRATEGY: {FT_TRAINING_LOGGING_STRATEGY}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - OUTPUT & REPORTING")
    print("=" * 80)
    print(f"OUTPUT_DIR: {FT_TRAINING_OUTPUT_DIR}")
    print(f"RUN_NAME: {FT_TRAINING_RUN_NAME}")
    print(f"REPORT_TO: {FT_TRAINING_REPORT_TO}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - REPRODUCIBILITY")
    print("=" * 80)
    print(f"SEED: {FT_TRAINING_SEED}")
    print(f"DATA_SEED: {FT_TRAINING_DATA_SEED}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - PERFORMANCE")
    print("=" * 80)
    print(f"DATALOADER_NUM_WORKERS: {FT_TRAINING_DATALOADER_NUM_WORKERS}")
    print(f"DATALOADER_PIN_MEMORY: {FT_TRAINING_DATALOADER_PIN_MEMORY}")
    print(f"DATALOADER_PREFETCH_FACTOR: {FT_TRAINING_DATALOADER_PREFETCH_FACTOR}")
    print(f"BATCH_SAMPLER: {FT_TRAINING_BATCH_SAMPLER}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - MIXED PRECISION")
    print("=" * 80)
    print(f"FP16: {FT_TRAINING_FP16}")
    print(f"BF16: {FT_TRAINING_BF16}")
    print(f"TF32: {FT_TRAINING_TF32}")
    print(f"FP16_FULL_EVAL: {FT_TRAINING_FP16_FULL_EVAL}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - MEMORY OPTIMIZATION")
    print("=" * 80)
    print(f"AUTO_FIND_BATCH_SIZE: {FT_TRAINING_AUTO_FIND_BATCH_SIZE}")
    print(f"GRADIENT_CHECKPOINTING: {FT_TRAINING_GRADIENT_CHECKPOINTING}")
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - ADVANCED")
    print("=" * 80)
    print(f"INCLUDE_INPUTS_FOR_METRICS: {FT_TRAINING_INCLUDE_INPUTS_FOR_METRICS}")
    print(f"LABEL_SMOOTHING_FACTOR: {FT_TRAINING_LABEL_SMOOTHING_FACTOR}")
    print(f"PREDICTION_LOSS_ONLY: {FT_TRAINING_PREDICTION_LOSS_ONLY}")
    
    print("\n" + "=" * 80)
    print("DATASET CONFIGURATION")
    print("=" * 80)
    print(f"DATASET_NAME: {FT_DATASET_DATASET_NAME}")
    print(f"DATASET_SPLIT: {FT_DATASET_DATASET_SPLIT}")
    print(f"TEST_SIZE: {FT_DATASET_TEST_SIZE}")
    print(f"ANCHOR_COLUMN: {FT_DATASET_ANCHOR_COLUMN}")
    print(f"POSITIVE_COLUMN: {FT_DATASET_POSITIVE_COLUMN}")
    print(f"SHUFFLE_TRAIN: {FT_DATASET_SHUFFLE_TRAIN}")
    print(f"SHUFFLE_SEED: {FT_DATASET_SHUFFLE_SEED}")
    print(f"MAX_TRAIN_SAMPLES: {FT_DATASET_MAX_TRAIN_SAMPLES}")
    print(f"MAX_EVAL_SAMPLES: {FT_DATASET_MAX_EVAL_SAMPLES}")
    print(f"CACHE_DIR: {FT_DATASET_CACHE_DIR}")
    print(f"KEEP_IN_MEMORY: {FT_DATASET_KEEP_IN_MEMORY}")
    
    print("\n" + "=" * 80)
    print("LOSS FUNCTION CONFIGURATION")
    print("=" * 80)
    print(f"TYPE: {FT_LOSS_TYPE}")
    print(f"MNRL_SCALE: {FT_LOSS_MNRL_SCALE}")
    print(f"MNRL_SIMILARITY_FCT: {FT_LOSS_MNRL_SIMILARITY_FCT}")
    print(f"COSINE_LOSS_FCT: {FT_LOSS_COSINE_LOSS_FCT}")
    
    print("\n" + "=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"EVALUATOR_TYPE: {FT_EVALUATION_EVALUATOR_TYPE}")
    print(f"IR_SCORE_FUNCTION: {FT_EVALUATION_IR_SCORE_FUNCTION}")
    print(f"IR_METRICS: {FT_EVALUATION_IR_METRICS}")
    print(f"IR_NAME: {FT_EVALUATION_IR_NAME}")
    print(f"IR_SAVE_CORPUS: {FT_EVALUATION_IR_SAVE_CORPUS}")
    print(f"IR_SAVE_QUERIES: {FT_EVALUATION_IR_SAVE_QUERIES}")
    print(f"SAVE_EVAL_DATA: {FT_EVALUATION_SAVE_EVAL_DATA}")
    print(f"EVAL_DATA_SUBDIR: {FT_EVALUATION_EVAL_DATA_SUBDIR}")
    
    print("\n" + "=" * 80)
    print("OUTPUT CONFIGURATION")
    print("=" * 80)
    print(f"CREATE_SUBDIRS: {FT_OUTPUT_CREATE_SUBDIRS}")
    print(f"SUBDIRS: {FT_OUTPUT_SUBDIRS}")
    print(f"LOGGING_LEVEL: {FT_OUTPUT_LOGGING_LEVEL}")
    print(f"LOGGING_SAVE_TO_FILE: {FT_OUTPUT_LOGGING_SAVE_TO_FILE}")
    print(f"LOGGING_LOG_FILENAME: {FT_OUTPUT_LOGGING_LOG_FILENAME}")
    print(f"LOGGING_CONSOLE_FORMAT: {FT_OUTPUT_LOGGING_CONSOLE_FORMAT}")
    print(f"LOGGING_CONSOLE_DATE_FORMAT: {FT_OUTPUT_LOGGING_CONSOLE_DATE_FORMAT}")
    print(f"LOGGING_FILE_FORMAT: {FT_OUTPUT_LOGGING_FILE_FORMAT}")
    print(f"SAVE_CONFIG: {FT_OUTPUT_SAVE_CONFIG}")
    print(f"CONFIG_FILENAME: {FT_OUTPUT_CONFIG_FILENAME}")
    print(f"SAVE_METRICS: {FT_OUTPUT_SAVE_METRICS}")
    print(f"METRICS_FILENAME: {FT_OUTPUT_METRICS_FILENAME}")
    print(f"SAVE_MODEL_INFO: {FT_OUTPUT_SAVE_MODEL_INFO}")
    print(f"MODEL_INFO_FILENAME: {FT_OUTPUT_MODEL_INFO_FILENAME}")
    
    print("\n" + "=" * 80)
    print("HARDWARE CONFIGURATION")
    print("=" * 80)
    print(f"CUDA_VISIBLE_DEVICES: {FT_HARDWARE_CUDA_VISIBLE_DEVICES}")
    print(f"USE_MULTI_GPU: {FT_HARDWARE_USE_MULTI_GPU}")
    print(f"LOCAL_RANK: {FT_HARDWARE_LOCAL_RANK}")
    print(f"EMPTY_CUDA_CACHE_STEPS: {FT_HARDWARE_EMPTY_CUDA_CACHE_STEPS}")
    print(f"MAX_MEMORY: {FT_HARDWARE_MAX_MEMORY}")
    print(f"NUM_THREADS: {FT_HARDWARE_NUM_THREADS}")
    
    print("\n" + "=" * 80)
    print("MONITORING AND DEBUGGING")
    print("=" * 80)
    print(f"SHOW_PROGRESS_BAR: {FT_MONITORING_SHOW_PROGRESS_BAR}")
    print(f"LOG_MEMORY_USAGE: {FT_MONITORING_LOG_MEMORY_USAGE}")
    print(f"MEMORY_LOG_INTERVAL: {FT_MONITORING_MEMORY_LOG_INTERVAL}")
    print(f"LOG_MODEL_STATS: {FT_MONITORING_LOG_MODEL_STATS}")
    print(f"MODEL_STATS_INTERVAL: {FT_MONITORING_MODEL_STATS_INTERVAL}")
    print(f"LOG_GRADIENT_NORM: {FT_MONITORING_LOG_GRADIENT_NORM}")
    print(f"EARLY_STOPPING_ENABLED: {FT_MONITORING_EARLY_STOPPING_ENABLED}")
    print(f"EARLY_STOPPING_PATIENCE: {FT_MONITORING_EARLY_STOPPING_PATIENCE}")
    print(f"EARLY_STOPPING_MIN_DELTA: {FT_MONITORING_EARLY_STOPPING_MIN_DELTA}")
    
    print("\n" + "=" * 80)
    print("POST-TRAINING ACTIONS")
    print("=" * 80)
    print(f"SAVE_FINAL_MODEL: {FT_POST_TRAINING_SAVE_FINAL_MODEL}")
    print(f"SAVE_MERGED_MODEL: {FT_POST_TRAINING_SAVE_MERGED_MODEL}")
    print(f"UPLOAD_TO_HUB: {FT_POST_TRAINING_UPLOAD_TO_HUB}")
    print(f"HUB_REPO_ID: {FT_POST_TRAINING_HUB_REPO_ID}")
    print(f"HUB_PRIVATE: {FT_POST_TRAINING_HUB_PRIVATE}")
    print(f"RUN_FINAL_EVALUATION: {FT_POST_TRAINING_RUN_FINAL_EVALUATION}")
    print(f"COMPARE_WITH_BASELINE: {FT_POST_TRAINING_COMPARE_WITH_BASELINE}")
    print(f"BASELINE_MODEL: {FT_POST_TRAINING_BASELINE_MODEL}")
    print(f"DELETE_CHECKPOINTS: {FT_POST_TRAINING_DELETE_CHECKPOINTS}")
    print(f"KEEP_BEST_CHECKPOINT: {FT_POST_TRAINING_KEEP_BEST_CHECKPOINT}")
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIONS - UNSLOTH")
    print("=" * 80)
    print(f"USE_UNSLOTH: {FT_ADVANCED_UNSLOTH_USE_UNSLOTH}")
    print(f"MAX_SEQ_LENGTH_OPTIMIZATION: {FT_ADVANCED_UNSLOTH_MAX_SEQ_LENGTH_OPTIMIZATION}")
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIONS - DISTRIBUTED")
    print("=" * 80)
    print(f"ENABLED: {FT_ADVANCED_DISTRIBUTED_ENABLED}")
    print(f"BACKEND: {FT_ADVANCED_DISTRIBUTED_BACKEND}")
    print(f"FIND_UNUSED_PARAMETERS: {FT_ADVANCED_DISTRIBUTED_FIND_UNUSED_PARAMETERS}")
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIONS - COMPILATION")
    print("=" * 80)
    print(f"TORCH_COMPILE: {FT_ADVANCED_TORCH_COMPILE}")
    print(f"TORCH_COMPILE_BACKEND: {FT_ADVANCED_TORCH_COMPILE_BACKEND}")
    print(f"TORCH_COMPILE_MODE: {FT_ADVANCED_TORCH_COMPILE_MODE}")
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIONS - PROFILING")
    print("=" * 80)
    print(f"ENABLED: {FT_ADVANCED_PROFILING_ENABLED}")
    print(f"PROFILE_MEMORY: {FT_ADVANCED_PROFILING_PROFILE_MEMORY}")
    print(f"PROFILE_STEPS: {FT_ADVANCED_PROFILING_PROFILE_STEPS}")
    print(f"OUTPUT_DIR: {FT_ADVANCED_PROFILING_OUTPUT_DIR}")
    
    print("\n" + "=" * 80)
    print("ADVANCED OPTIONS - DEBUG")
    print("=" * 80)
    print(f"ENABLED: {FT_ADVANCED_DEBUG_ENABLED}")
    print(f"DEBUG_OVERFLOW: {FT_ADVANCED_DEBUG_DEBUG_OVERFLOW}")
    print(f"DEBUG_UNDERFLOW: {FT_ADVANCED_DEBUG_DEBUG_UNDERFLOW}")