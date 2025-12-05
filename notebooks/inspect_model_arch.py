"""
Script to inspect model architecture and find correct LoRA target modules
"""
from transformers import AutoModel
import torch

def inspect_model_architecture(model_id: str):
    """
    Inspect model architecture to find attention module names
    
    Args:
        model_id: HuggingFace model identifier
    """
    print(f"Loading model: {model_id}")
    print("=" * 80)
    
    # Load model
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("\nModel Architecture:")
    print("=" * 80)
    print(model)
    
    print("\n\nAll Named Modules:")
    print("=" * 80)
    
    # Collect unique module types
    module_types = set()
    attention_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_types.add(module_type)
        
        # Look for attention-related modules
        if any(keyword in name.lower() for keyword in ['attention', 'attn', 'query', 'key', 'value', 'q_', 'k_', 'v_', 'o_']):
            if 'Linear' in module_type or 'Conv' in module_type:
                attention_modules.append((name, module_type))
    
    print("\nAttention-Related Linear Layers:")
    print("-" * 80)
    for name, module_type in sorted(attention_modules):
        print(f"{name:60s} -> {module_type}")
    
    print("\n\nRecommended LoRA Target Modules:")
    print("=" * 80)
    
    # Extract common patterns
    target_patterns = set()
    for name, _ in attention_modules:
        # Get the last part of the module name
        parts = name.split('.')
        if len(parts) > 0:
            last_part = parts[-1]
            if any(keyword in last_part for keyword in ['query', 'key', 'value', 'dense', 'q', 'k', 'v', 'o']):
                target_patterns.add(last_part)
    
    print("\nBased on the architecture, use these target modules:")
    print("-" * 80)
    for pattern in sorted(target_patterns):
        print(f"  - \"{pattern}\"")
    
    print("\n\nYAML Configuration:")
    print("=" * 80)
    print("lora:")
    print("  target_modules:")
    for pattern in sorted(target_patterns):
        print(f"    - \"{pattern}\"")
    
    print("\n\nAll Unique Module Types Found:")
    print("=" * 80)
    for module_type in sorted(module_types):
        print(f"  - {module_type}")


if __name__ == "__main__":
    # Inspect mixedbread model
    model_id = "mixedbread-ai/mxbai-embed-large-v1"
    
    try:
        inspect_model_architecture(model_id)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTrying alternative inspection method...")
        
        # Alternative: Just load config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"\nModel Config:")
        print(config)