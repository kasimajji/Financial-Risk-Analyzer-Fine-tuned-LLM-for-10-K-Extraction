# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "mistralai/Mistral-7B-v0.1"
adapter_path = "/content/drive/MyDrive/GitHub/finllm/model/qlora_adapters"
offload_dir = "/content/drive/MyDrive/GitHub/finllm/offload"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ✅ Load adapter with offload_dir here instead
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    device_map="auto",
    offload_folder=offload_dir  # <- ✅ THIS is the key fix
)

#Inference
prompt = """Based on the 10-K filing below, identify the risk in structured format.

Filing:
"In the past fiscal year, the company experienced multiple attempted breaches into its internal IT infrastructure, and continues to face persistent cybersecurity threats targeting customer data."

Respond in JSON:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("---------Generated Output---------")
print(response.split("Respond in JSON:")[-1].strip())
