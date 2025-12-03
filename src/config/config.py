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