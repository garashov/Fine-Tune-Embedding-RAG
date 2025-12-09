import os
import subprocess
import sys

# ----------------------------
# CONFIG
# ----------------------------
# Path to your merged HuggingFace model
hf_model_path = r"C:\Users\elnur\Desktop\Python\Projects\AIVB\Fine Tuning\Embedding\Unsloth\data\merged_models\20251209_102938\ft_custom_mxbai-embed-large-v1_embedding"

# Name of the GGUF output file
gguf_filename = "mxbai-embed-custom.gguf"
gguf_path = os.path.join(hf_model_path, gguf_filename)

# Path where llama.cpp will be cloned
llama_cpp_path = r"C:\Users\elnur\Desktop\Python\Projects\AIVB\Fine Tuning\Embedding\Unsloth\data\llama.cpp"

# ----------------------------
# STEP 1: Clone llama.cpp if it doesn't exist
# ----------------------------
if not os.path.exists(llama_cpp_path):
    print(f"Cloning llama.cpp into {llama_cpp_path}...")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_path], check=True)
else:
    print(f"llama.cpp already exists at {llama_cpp_path}")

# ----------------------------
# STEP 2: Run convert_hf_to_gguf.py
# ----------------------------
convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")

if not os.path.exists(convert_script):
    print(f"ERROR: convert_hf_to_gguf.py not found in {llama_cpp_path}")
    sys.exit(1)

print(f"Converting HF model to GGUF at {gguf_path} ...")

subprocess.run([
    sys.executable,
    convert_script,
    hf_model_path,
    "--outfile",
    gguf_path
], check=True)

print("✅ Conversion complete!")
print(f"GGUF file saved at: {gguf_path}")
