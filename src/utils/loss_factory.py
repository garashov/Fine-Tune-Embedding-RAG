import logging
from typing import Optional, Any
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss, CosineSimilarityLoss

from src.config.config import (
    FT_LOSS_TYPE,
    FT_LOSS_MNRL_SCALE,
    FT_LOSS_MNRL_SIMILARITY_FCT,
)


class LossFactory:
    """
    Creates loss functions for training
    Can be moved to core/loss_factory.py
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize loss factory"""
        self.logger = logger or logging.getLogger(__name__)
    
    def create_loss(self, model: SentenceTransformer) -> Any:
        """
        Create loss function based on configuration
        
        Args:
            model: SentenceTransformer model
        
        Returns:
            Configured loss function
        """
        self.logger.info("=" * 80)
        self.logger.info("CREATING LOSS FUNCTION")
        self.logger.info("=" * 80)
        self.logger.info(f"Loss type: {FT_LOSS_TYPE}")
        
        try:
            if FT_LOSS_TYPE == "MultipleNegativesRankingLoss":
                loss = MultipleNegativesRankingLoss(
                    model=model,
                    scale=FT_LOSS_MNRL_SCALE,
                    similarity_fct=cos_sim if FT_LOSS_MNRL_SIMILARITY_FCT == "cos_sim" else None
                )
                self.logger.info(f"[OK] MultipleNegativesRankingLoss created (scale={FT_LOSS_MNRL_SCALE})")
                
            elif FT_LOSS_TYPE == "CosineSimilarityLoss":
                loss = CosineSimilarityLoss(model=model)
                self.logger.info(f"[OK] CosineSimilarityLoss created")
                
            else:
                raise ValueError(f"Unsupported loss type: {FT_LOSS_TYPE}")
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Failed to create loss function: {str(e)}", exc_info=True)
            raise

