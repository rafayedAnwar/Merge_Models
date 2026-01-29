import torch
from safetensors.torch import load_file, save_file

# --- CONFIGURATION ---
model_A_path = "./models/SST-2__InsertSent__BERT/model.safetensors"
model_B_path = "./models/SST-2__Clean__BERT/model.safetensors"
output_path = "./models/SST-2__InsertSent_Merged_Model/model.safetensors"
alpha = 0.5  # Weight for averaging. 0.5 means (A + B) / 2

# 1. Load the models
print("Loading models...")
state_dict_A = load_file(model_A_path)
state_dict_B = load_file(model_B_path)
merged_state_dict = {}

# 2. Define the Pattern Logic
# Cycle: A -> Merge -> B -> Merge
def get_source_type(layer_idx):
    cycle = layer_idx % 4
    if cycle == 0: return "A"       # e.g., Layer 0, 4, 8
    if cycle == 1: return "Merge"   # e.g., Layer 1, 5, 9
    if cycle == 2: return "B"       # e.g., Layer 2, 6, 10
    if cycle == 3: return "Merge"   # e.g., Layer 3, 7, 11

# 3. Iterate through all keys
all_keys = state_dict_A.keys()

for key in all_keys:
    # Check if this key belongs to a specific encoder layer
    if "encoder.layer." in key:
        # Extract layer number (e.g., "bert.encoder.layer.5.output..." -> 5)
        layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
        
        action = get_source_type(layer_num)
        
        if action == "A":
            # 90% from model A (temp wag)
            merged_state_dict[key] = 0.5 * state_dict_A[key] + 0.5 * state_dict_B[key]
        elif action == "B":
            # 90% from model B (temp wag)
            merged_state_dict[key] = 0.5 * state_dict_B[key] + 0.5 * state_dict_A[key]
        elif action == "Merge":
            # WAG: Weighted Average
            merged_state_dict[key] = (.5 * state_dict_A[key]) + (.5 * state_dict_B[key])

    else:
        # NON-LAYER KEYS (Embeddings, Pooler, Classification Head)
        # Strategy: Average them to maintain compatibility with both A and B layers
        tensor_a = state_dict_A[key]
        tensor_b = state_dict_B[key]
        merged_state_dict[key] = (tensor_a + tensor_b) / 2.0

# 4. Save
print(f"Saving merged model to {output_path}...")
save_file(merged_state_dict, output_path)
print("Done! Don't forget to copy config.json and tokenizer files to the new folder.")