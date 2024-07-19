import os
import yaml
from pathlib import Path

# Avoid duplicate OpenMP runtime conflict.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Specify the data_folder
data_folder = Path.home() / "Data"
if not data_folder.exists():
    raise EnvironmentError(f"{data_folder} does not exist.")

# Load API keys
with open(data_folder / "api/api_keys_for_ai.yaml", "r") as keys_file:
    keys = yaml.safe_load(keys_file)

# OpenAI api pricing
with open("openai_api_pricing.yaml", "r") as file:
    openai_api_pricing = yaml.safe_load(file)
