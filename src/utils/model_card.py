import logging
from pathlib import Path
from typing import Optional, Dict, List

from src.config.config import (
    HF_PUSH_METADATA_DESCRIPTION,
    HF_PUSH_METADATA_TAGS,
    HF_PUSH_METADATA_LANGUAGE,
    HF_PUSH_METADATA_LICENSE,
    HF_PUSH_METADATA_DATASETS,
    HF_PUSH_METADATA_BASE_MODEL,
    HF_PUSH_MODEL_CARD_GENERATE_CARD,
    HF_PUSH_MODEL_CARD_TEMPLATE,
    HF_PUSH_MODEL_CARD_INCLUDE_SECTIONS,
    HF_PUSH_MODEL_CARD_CUSTOM_CONTENT,
)

class ModelCardGenerator:
    """
    Generates model cards for HuggingFace Hub
    Can be moved to utils/model_card.py
    """
    
    def __init__(
        self,
        generate_card: bool = HF_PUSH_MODEL_CARD_GENERATE_CARD,
        template: str = HF_PUSH_MODEL_CARD_TEMPLATE,
        include_sections: List[str] = HF_PUSH_MODEL_CARD_INCLUDE_SECTIONS,
        custom_content: Dict = HF_PUSH_MODEL_CARD_CUSTOM_CONTENT,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize model card generator"""
        self.generate_card = generate_card
        self.template = template
        self.include_sections = include_sections
        self.custom_content = custom_content
        self.logger = logger or logging.getLogger(__name__)
    
    def generate(
        self,
        model_path: Path,
        repo_id: str
    ) -> Optional[str]:
        """
        Generate model card content
        
        Args:
            model_path: Path to model
            repo_id: Repository ID
        
        Returns:
            Model card markdown content or None
        """
        if not self.generate_card:
            return None
        
        self.logger.info("Generating model card...")
        
        if self.template == "minimal":
            return self._generate_minimal_card(repo_id)
        elif self.template == "custom":
            return self._generate_custom_card()
        else:
            return self._generate_default_card(repo_id)
    
    def _generate_minimal_card(self, repo_id: str) -> str:
        """Generate minimal model card"""
        return f"""---
library_name: sentence-transformers
tags:
{self._format_tags()}
---

# {repo_id}

This is a fine-tuned version of {HF_PUSH_METADATA_BASE_MODEL}.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{repo_id}')
sentences = ['Example sentence 1', 'Example sentence 2']
embeddings = model.encode(sentences)
```
"""
    
    def _generate_default_card(self, repo_id: str) -> str:
        """Generate default comprehensive model card"""
        sections = []
        
        # Header with metadata
        sections.append(f"""---
library_name: sentence-transformers
language:
{self._format_list(HF_PUSH_METADATA_LANGUAGE)}
license: {HF_PUSH_METADATA_LICENSE}
tags:
{self._format_tags()}
base_model: {HF_PUSH_METADATA_BASE_MODEL}
datasets:
{self._format_list(HF_PUSH_METADATA_DATASETS)}
---

# {repo_id}

{HF_PUSH_METADATA_DESCRIPTION}
""")
        
        # Model description
        if "model_description" in self.include_sections:
            sections.append("""
## Model Description

This model is a fine-tuned version of {base_model}, optimized for specific embedding tasks.

**Base Model:** {base_model}
**Fine-tuning Datasets:** {datasets}
""".format(
                base_model=HF_PUSH_METADATA_BASE_MODEL,
                datasets=", ".join(HF_PUSH_METADATA_DATASETS)
            ))
        
        # Usage
        sections.append(f"""
## Usage

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('{repo_id}')

# Generate embeddings
sentences = ['Example sentence 1', 'Example sentence 2']
embeddings = model.encode(sentences)

# Compute similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
```
""")
        
        # Intended uses
        if "intended_uses" in self.include_sections:
            sections.append("""
## Intended Uses

This model is designed for:
- Semantic search and information retrieval
- Document similarity and clustering
- Question-answering systems
- Recommendation systems
""")
        
        # Limitations
        if "limitations" in self.include_sections:
            sections.append("""
## Limitations

- Performance may vary on domains different from the fine-tuning data
- Best results with text similar in length and style to training examples
- Inherited biases from base model and training data
""")
        
        return "\n".join(sections)
    
    def _generate_custom_card(self) -> str:
        """Generate custom model card from provided content"""
        sections = []
        
        # Add header
        sections.append(f"""---
library_name: sentence-transformers
tags:
{self._format_tags()}
---
""")
        
        # Add custom sections
        for key, value in self.custom_content.items():
            if value:
                title = key.replace('_', ' ').title()
                sections.append(f"\n## {title}\n\n{value}")
        
        return "\n".join(sections)
    
    def _format_tags(self) -> str:
        """Format tags for YAML front matter"""
        return "\n".join([f"- {tag}" for tag in HF_PUSH_METADATA_TAGS])
    
    def _format_list(self, items: List[str]) -> str:
        """Format list for YAML front matter"""
        return "\n".join([f"- {item}" for item in items])
