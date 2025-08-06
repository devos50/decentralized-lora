# minimal_in_memory_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1) Load a tiny model & tokenizer (GPU/CPU ok) ---
base_id = "sshleifer/tiny-gpt2"  # small and fast for demos
base = AutoModelForCausalLM.from_pretrained(base_id).to(DEVICE)

# --- 2) Wrap with PEFT once, then add per-client adapters on the fly ---
# Create a "shell" PEFT model (we'll ignore the default adapter)
lora_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=8, lora_alpha=16, lora_dropout=0.05,
    # for GPT-2-style blocks, these are common linear modules:
    target_modules=["c_attn", "c_fc", "c_proj"],
)
model: PeftModel = get_peft_model(base, lora_cfg).to(DEVICE)
model.eval()

for adapter_name in ["client_a", "client_b", "client_c"]:
    if adapter_name not in model.peft_config:
        model.add_adapter(adapter_name, lora_cfg)
    model.set_adapter(adapter_name)


# Merge them in-memory into a new adapter called "global"
model.add_weighted_adapter(
    adapters=["client_a", "client_b", "client_c"],
    weights=[0.6, 0.2, 0.2],              # e.g., token-count weights
    adapter_name="global",
    combination_type="linear",        # or "ties", "dare_ties", "svd" (if supported by your PEFT version)
)
model.set_adapter("global")

output_state_dict = get_peft_model_state_dict(model, adapter_name="global")
print(output_state_dict.keys())
print(output_state_dict["base_model.model.transformer.h.0.attn.c_attn.lora_A.weight"])

# # Serialize the global adapter to disk
# model.save_pretrained(
#     "adapters/global",
#     selected_adapters=["global"],   # save just this adapter
#     safe_serialization=True           # writes adapter_model.safetensors
# )