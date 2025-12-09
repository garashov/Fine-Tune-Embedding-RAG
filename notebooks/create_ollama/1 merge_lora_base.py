import os
from peft import PeftModel
from datetime import datetime 
from transformers import AutoModel, AutoTokenizer

# Constants
base = "mixedbread-ai/mxbai-embed-large-v1"
lora = r"C:\Users\elnur\Desktop\Python\Projects\AIVB\Fine Tuning\Embedding\Unsloth\data\fine_tuning\20251205_160130\ft_custom_mxbai-embed-large-v1_embedding"
timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"C:\Users\elnur\Desktop\Python\Projects\AIVB\Fine Tuning\Embedding\Unsloth\data\merged_models\{timestamp}\ft_custom_mxbai-embed-large-v1_embedding"

# Create output directory
os.makedirs(output_dir, exist_ok=False)

# Merge LoRA weights into the base model
model = AutoModel.from_pretrained(base)
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()     # important: merge LoRA→full model

# Save the merged model and tokenizer
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.save_pretrained(output_dir)
