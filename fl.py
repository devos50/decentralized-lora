#!/usr/bin/env python
# federated_lora.py

import argparse, json, os, math, itertools
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, PeftModel


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_clients(spec: str) -> List[Dict]:
    """
    Accept either:
      - path to a JSON file: [{"id":"c0","data":"path_or_hf_name","weight":12345}, ...]
      - or a comma-separated CLI list: id:data:weight,id:data:weight
    """
    if os.path.isfile(spec):
        with open(spec) as f:
            return json.load(f)
    clients = []
    for item in spec.split(","):
        cid, data, weight = item.split(":")
        clients.append({"id": cid, "data": data, "weight": float(weight)})
    return clients

def load_text_dataset(data_spec: str, tokenizer, block_size=1024):
    """
    data_spec can be:
      - a local file path to .txt or .jsonl with "text" field
      - a HF dataset name like "imdb" (expects a 'text' column)
    """
    if os.path.isfile(data_spec) and data_spec.endswith(".txt"):
        ds = load_dataset("text", data_files=data_spec)
    else:
        # Try json/jsonl or HF repo
        try:
            ds = load_dataset("json", data_files=data_spec)
        except Exception:
            ds = load_dataset(data_spec)

    # Find a text column
    col = "text"
    if col not in ds["train"].column_names:
        # pick the first string column heuristically
        for c in ds["train"].column_names:
            if isinstance(ds["train"][0][c], str):
                col = c
                break

    def tok(batch):
        return tokenizer(batch[col])

    tokenized = ds.map(tok, batched=True, remove_columns=[c for c in ds["train"].column_names if c != col])
    # group texts into blocks
    def group_texts(examples):
        concatenated = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_len = (len(concatenated) // block_size) * block_size
        result = {
            "input_ids": [concatenated[i:i+block_size] for i in range(0, total_len, block_size)],
        }
        result["labels"] = [ids.copy() for ids in result["input_ids"]]
        return result

    return tokenized.map(group_texts, batched=True, remove_columns=tokenized["train"].column_names)


# ---------------------------
# Training one client adapter
# ---------------------------

def train_one_client(
    model: PeftModel,
    tokenizer,
    client_id: str,
    data_spec: str,
    out_root: str,
    lora_cfg: LoraConfig,
    epochs: int,
    per_device_train_batch_size: int,
    lr: float,
    gradient_accumulation_steps: int,
    logging_steps: int,
    fp16: bool,
    bf16: bool,
):
    """
    Reuses the SAME base model in memory. We add (or load) a uniquely named adapter for this client,
    activate it, train only that adapter, then save just that adapter.
    """
    # If this adapter name does not exist yet on the model, create it.
    # (We intentionally do not touch previously trained adapters.)
    if client_id not in model.peft_config:
        model.add_adapter(client_id, lora_cfg)

    model.set_adapter(client_id)
    model.train()

    ds = load_text_dataset(data_spec, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    out_dir = os.path.join(out_root, client_id)
    ensure_dir(out_dir)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_strategy="no",
        fp16=fp16,
        bf16=bf16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Save ONLY this client's adapter
    model.save_pretrained(out_dir, selected_adapters=[client_id])

    return out_dir


# ---------------------------
# Aggregation (server)
# ---------------------------

def aggregate_adapters(
    base_model_id: str,
    client_adapter_dirs: List[str],
    client_names: List[str],
    weights: List[float],
    out_dir: str,
    device_map="auto",
    combination_type: str = "linear",  # alternatives: "ties", "dare_ties", "svd"
):
    base = AutoModelForCausalLM.from_pretrained(base_model_id, device_map=device_map)
    model = PeftModel.from_pretrained(base, client_adapter_dirs[0], adapter_name=client_names[0])
    for d, name in zip(client_adapter_dirs[1:], client_names[1:]):
        model.load_adapter(d, adapter_name=name)

    # Normalize weights
    s = sum(weights)
    if s <= 0:
        raise ValueError("Sum of weights must be > 0")
    w = [wi / s for wi in weights]

    model.add_weighted_adapter(client_names, w, adapter_name="global", combination_type=combination_type)
    model.set_adapter("global")

    ensure_dir(out_dir)
    model.save_pretrained(out_dir)  # saves the merged adapter as "global"

    return out_dir


# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Simulate decentralized LoRA with one base model and multiple adapters.")
    # What to do
    p.add_argument("--do", choices=["round", "clients", "aggregate"], default="round",
                   help="round = train all clients then aggregate; clients = only train clients; aggregate = only aggregate")
    # Base & tokenizer
    p.add_argument("--base-model", required=True)
    p.add_argument("--tokenizer", default=None, help="Defaults to --base-model if not set")
    p.add_argument("--device-map", default="auto")
    # Clients
    p.add_argument("--clients", required=True,
                   help="Path to JSON array or 'id:data:weight,id:data:weight,...'")
    p.add_argument("--clients-out", default="adapters/clients")
    # LoRA config
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear",
                   help='e.g. "all-linear" or comma list like "q_proj,k_proj,v_proj,o_proj"')
    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    # Aggregation
    p.add_argument("--global-out", default="adapters/global_round")
    p.add_argument("--combination", default="linear", choices=["linear", "ties", "dare_ties", "svd"])
    # Advanced
    p.add_argument("--init-global-adapter", default=None,
                   help="Optional path to a global adapter to initialize each client (for rounds > 1).")
    p.add_argument("--bake", action="store_true",
                   help="After aggregation, also bake the global adapter into base weights for inference.")
    args = p.parse_args()

    tok_id = args.tokenizer or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Single base model reused in memory for all clients
    base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=args.device_map)

    # Build one PeftModel wrapper (we'll add/load adapters for each client onto this shared base)
    # If we have an initial global adapter, load it so its LoRA layers are present;
    # otherwise create a shell adapter we won't train/use.
    if args.init_global_adapter:
        model = PeftModel.from_pretrained(base, args.init_global_adapter, adapter_name="global_init", is_trainable=False)
    else:
        lora_cfg_init = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=None if args.target_modules == "all-linear" else args.target_modules.split(","),
        )
        model = get_peft_model(base, lora_cfg_init)  # creates a 'default' adapter; we'll ignore it.

    # Prepare the LoRA config we'll use for each client
    lora_cfg_clients = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=None if args.target_modules == "all-linear" else args.target_modules.split(","),
    )

    clients = read_clients(args.clients)
    client_ids = [c["id"] for c in clients]
    client_weights = [float(c.get("weight", 1.0)) for c in clients]
    client_adapter_dirs = []

    if args.do in ("round", "clients"):
        ensure_dir(args.clients_out)

        for c in clients:
            cid, data = c["id"], c["data"]

            # If we want to start each client from a prior global adapter (round > 1),
            # load that adapter *into this client's name* so it becomes the initial weights.
            if args.init_global_adapter and cid not in model.peft_config:
                model.load_adapter(args.init_global_adapter, adapter_name=cid, is_trainable=True)
            # Otherwise, add a fresh adapter for this client
            if cid not in model.peft_config:
                model.add_adapter(cid, lora_cfg_clients)

            out_dir = train_one_client(
                model=model,
                tokenizer=tokenizer,
                client_id=cid,
                data_spec=data,
                out_root=args.clients_out,
                lora_cfg=lora_cfg_clients,
                epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                lr=args.lr,
                gradient_accumulation_steps=args.grad_accum,
                logging_steps=args.logging_steps,
                fp16=args.fp16,
                bf16=args.bf16,
            )
            client_adapter_dirs.append(out_dir)

        # After training all clients, it's a good idea to move the active adapter off
        # to avoid accidental updates if you keep the process alive.
        if client_ids:
            model.set_adapter(client_ids[-1])
            model.eval()

    if args.do in ("round", "aggregate"):
        if not client_adapter_dirs:
            # Collect client adapter directories from --clients_out if we didn't just train
            client_adapter_dirs = [os.path.join(args.clients_out, cid) for cid in client_ids]
        ensure_dir(args.global_out)

        merged_dir = aggregate_adapters(
            base_model_id=args.base_model,
            client_adapter_dirs=client_adapter_dirs,
            client_names=client_ids,
            weights=client_weights,
            out_dir=args.global_out,
            device_map=args.device_map,
            combination_type=args.combination,
        )
        print(f"[server] saved merged global adapter to: {merged_dir}")

        if args.bake:
            # Optional: bake adapter into base for single-file inference
            base2 = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=args.device_map)
            model2 = PeftModel.from_pretrained(base2, merged_dir, adapter_name="global")
            model2.set_adapter("global")
            merged_base = model2.merge_and_unload()
            baked_dir = os.path.join(args.global_out, "baked_base")
            ensure_dir(baked_dir)
            merged_base.save_pretrained(baked_dir)
            print(f"[server] baked merged weights to: {baked_dir}")


if __name__ == "__main__":
    main()