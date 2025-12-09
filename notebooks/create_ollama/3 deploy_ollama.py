import os
import json
import requests
import subprocess

# ----------------------------
# CONFIG
# ----------------------------
# Folder containing your GGUF + tokenizer files
model_folder = r"C:\Users\elnur\Desktop\Python\Projects\AIVB\Fine Tuning\Embedding\Unsloth\data\merged_models\20251209_102938\ft_custom_mxbai-embed-large-v1_embedding"
gguf_filename = "mxbai-embed-custom.gguf"
ollama_model_name = "mxbai-embed-custom"

gguf_path = os.path.join(model_folder, gguf_filename)
modelfile_path = os.path.join(model_folder, "Modelfile")

# ----------------------------
# STEP 1: Check if Ollama model exists
# ----------------------------
try:
    result = subprocess.run(
        ["ollama", "list", "--json"],
        capture_output=True, text=True, check=True
    )
    models = json.loads(result.stdout)
    model_names = [m["name"] for m in models]
except Exception as e:
    print("Error fetching Ollama models:", e)
    models = []
    model_names = []

if ollama_model_name in model_names:
    print(f"Ollama model '{ollama_model_name}' already exists.")
else:
    print(f"Ollama model '{ollama_model_name}' not found. Creating it...")

    # ----------------------------
    # STEP 2: Create Modelfile
    # ----------------------------
    with open(modelfile_path, "w") as f:
        f.write(f"from ./{gguf_filename}\n")
    print("Modelfile created.")

    # ----------------------------
    # STEP 3: Create Ollama model
    # ----------------------------
    subprocess.run(["ollama", "create", ollama_model_name, "-f", modelfile_path], check=True)
    print(f"Ollama model '{ollama_model_name}' created successfully.")

# ----------------------------
# STEP 4: Test embedding
# ----------------------------
def get_ollama_embedding(text: str, model: str = ollama_model_name):
    """Get embeddings from Ollama API"""
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


test_text = "Hello world"
embedding_vector = get_ollama_embedding(test_text)
print(f"Embedding vector dimension: {len(embedding_vector)}")
print(f"First 10 values: {embedding_vector[:10]}")