
import unsloth                  
from unsloth import FastModel

import logging
from peft import TaskType
from datetime import datetime
from transformers import AutoModel
from typing import Optional, Tuple, Any
import sentence_transformers
from sentence_transformers import SentenceTransformer

from src.utils.data_models import ModelInfo
from src.config.config import (
    FT_MODEL_BASE_MODEL_ID,
    FT_MODEL_MAX_SEQ_LENGTH,
    FT_MODEL_LOAD_IN_4BIT,
    FT_MODEL_TRUST_REMOTE_CODE,
    FT_MODEL_DTYPE,
    FT_LORA_R,
    FT_LORA_ALPHA,
    FT_LORA_DROPOUT,
    FT_LORA_USE_RSLORA,
    FT_LORA_TARGET_MODULES,
    FT_LORA_EXCLUDE_MODULES,
    FT_LORA_BIAS,
    FT_LORA_USE_GRADIENT_CHECKPOINTING,
    FT_LORA_MODULES_TO_SAVE,
)


class ModelLoader:
    """
    Handles model loading with Unsloth optimizations
    Can be moved to core/model_loader.py
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize model loader"""
        self.logger = logger or logging.getLogger(__name__)
    
    def load_base_model(self) -> Tuple[Any, Any]:
        """
        Load the base Qwen3 embedding model with Unsloth optimizations
        
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("LOADING BASE MODEL")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {FT_MODEL_BASE_MODEL_ID}")
        self.logger.info(f"Max sequence length: {FT_MODEL_MAX_SEQ_LENGTH}")
        self.logger.info(f"Load in 4-bit: {FT_MODEL_LOAD_IN_4BIT}")
        
        try:
            model, tokenizer = FastModel.from_pretrained(
                model_name=FT_MODEL_BASE_MODEL_ID,
                auto_model=AutoModel,
                max_seq_length=FT_MODEL_MAX_SEQ_LENGTH,
                dtype=FT_MODEL_DTYPE,
                load_in_4bit=FT_MODEL_LOAD_IN_4BIT,
                trust_remote_code=FT_MODEL_TRUST_REMOTE_CODE,
            )
            
            self.logger.info(f"[OK] Model loaded successfully")
            self.logger.info(f"  Model dtype: {model.dtype}")
            self.logger.info(f"  Model device: {model.device}")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def attach_lora_adapters(self, model: Any) -> Any:
        """
        Attach QLoRA adapters to the base model
        
        Args:
            model: Base model
        
        Returns:
            Model with LoRA adapters attached
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ATTACHING LORA ADAPTERS")
        self.logger.info("=" * 80)
        self.logger.info(f"LoRA rank: {FT_LORA_R}")
        self.logger.info(f"LoRA alpha: {FT_LORA_ALPHA}")
        self.logger.info(f"LoRA dropout: {FT_LORA_DROPOUT}")
        self.logger.info(f"Target modules: {FT_LORA_TARGET_MODULES}")
        
        try:
            model = FastModel.get_peft_model(
                model,
                r=FT_LORA_R,
                lora_alpha=FT_LORA_ALPHA,
                lora_dropout=FT_LORA_DROPOUT,
                target_modules=FT_LORA_TARGET_MODULES,
                exclude_modules=FT_LORA_EXCLUDE_MODULES,
                use_rslora=FT_LORA_USE_RSLORA,
                bias=FT_LORA_BIAS,
                use_gradient_checkpointing=FT_LORA_USE_GRADIENT_CHECKPOINTING,
                modules_to_save=FT_LORA_MODULES_TO_SAVE,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            # Calculate parameter statistics
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_percentage = 100 * trainable_params / total_params
            
            self.logger.info(f"[OK] LoRA adapters attached")
            self.logger.info(f"  Trainable parameters: {trainable_params:,}")
            self.logger.info(f"  Total parameters: {total_params:,}")
            self.logger.info(f"  Trainable percentage: {trainable_percentage:.4f}%")
            
            # Store info for later
            self.model_info = ModelInfo(
                base_model=FT_MODEL_BASE_MODEL_ID,
                total_parameters=total_params,
                trainable_parameters=trainable_params,
                trainable_percentage=trainable_percentage,
                lora_rank=FT_LORA_R,
                lora_alpha=FT_LORA_ALPHA,
                target_modules=FT_LORA_TARGET_MODULES,
                timestamp=datetime.now().isoformat(),
                device=str(model.device)
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to attach LoRA adapters: {str(e)}", exc_info=True)
            raise
    
    def create_sentence_transformer(
        self,
        model: Any,
        tokenizer: Any
    ) -> SentenceTransformer:
        """
        Wrap the Unsloth model in a SentenceTransformer
        
        Args:
            model: Unsloth model with LoRA adapters
            tokenizer: Model tokenizer
        
        Returns:
            SentenceTransformer model ready for training
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CREATING SENTENCE TRANSFORMER WRAPPER")
        self.logger.info("=" * 80)
        
        try:
            # Create Transformer module
            transformer_module = sentence_transformers.models.Transformer(
                model_name_or_path=FT_MODEL_BASE_MODEL_ID,
                max_seq_length=FT_MODEL_MAX_SEQ_LENGTH,
            )
            
            # Replace with our LoRA-patched model
            transformer_module.auto_model = model
            transformer_module.tokenizer = tokenizer
            
            self.logger.info("[OK] Assigned Unsloth LoRA model to Transformer module")
            
            # Create Pooling module
            hidden_size = model.config.hidden_size
            pooling_module = sentence_transformers.models.Pooling(
                word_embedding_dimension=hidden_size,
                pooling_mode="mean",
            )
            self.logger.info(f"[OK] Created Pooling module (mean pooling, hidden_size={hidden_size})")
            
            # Create Normalize module
            normalize_module = sentence_transformers.models.Normalize()
            
            # Assemble modules
            modules = [transformer_module, pooling_module, normalize_module]
            
            # Create SentenceTransformer
            sbert_model = SentenceTransformer(modules=modules)
            
            self.logger.info("[OK] SentenceTransformer wrapper created successfully")
            
            return sbert_model
            
        except Exception as e:
            self.logger.error(f"Failed to create SentenceTransformer: {str(e)}", exc_info=True)
            raise
