"""
Unified Configuration Loader
Loads and merges multiple YAML configuration files
"""

import os
from envyaml import EnvYAML



# Initialize the unified configuration
_current_dir = os.path.dirname(__file__)
_yamls_dir = os.path.join(_current_dir, "yamls")

# List all config files to load (order matters - later files override earlier ones)
CONFIG_FILES = [
    os.path.join(_yamls_dir, "evaluation.yaml"),
    os.path.join(_yamls_dir, "fine_tuning.yaml"),
    os.path.join(_yamls_dir, "hf_push.yaml"),
]


EVAL_CONFIG = EnvYAML(CONFIG_FILES[0], strict=False)
FT_CONFIG = EnvYAML(CONFIG_FILES[1], strict=False)
HF_PUSH_CONFIG = EnvYAML(CONFIG_FILES[2], strict=False)