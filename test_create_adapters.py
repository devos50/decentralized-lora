from typing import Dict
import datasets
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForSequenceClassification

USERS = 12

dataset = datasets.load_dataset("ag_news", cache_dir="data/datasets")

num_labels = dataset['train'].features['label'].num_classes
class_names = dataset["train"].features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
# We will need this for our classifier.
id2label = {i: label for i, label in enumerate(class_names)}

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", id2label=id2label, cache_dir="models")
peft_model = get_peft_model(base_model, peft_config, adapter_name="global")

# Create adapters for each user
for adapter_name in ["client_%d" % i for i in range(USERS)]:
    if adapter_name not in peft_model.peft_config:
        peft_model.add_adapter(adapter_name, peft_config)
        adapter: Dict = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
        print(adapter.keys())