"""
LoRA fine-tuning script for the character LLM.

Uses QLoRA (4-bit quantization) so it runs on a single A100 (40GB)
or two RTX 4090s. No full fine-tune needed — LoRA adapter is ~200MB
and makes the model permanently behave as your character.

Run:
  python finetune/train.py

Hardware requirements:
  Minimum:  1× RTX 3090 (24GB)  — use --model 8b
  Good:     1× A100 (40GB)      — use --model 70b (recommended)
  Ideal:    2× A100 (80GB)      — use --model 70b, faster

Cloud options (cost ~$50-150 for full run):
  RunPod:   https://runpod.io  (A100 ~$2/hr)
  Vast.ai:  https://vast.ai   (A100 ~$1.5/hr, cheaper)
  Lambda:   https://lambda.ai  (A100 ~$1.1/hr)
"""
import argparse
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("train")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

# Default model — Llama 3.3 70B Instruct (best balance of quality vs cost)
DEFAULT_MODEL  = "meta-llama/Llama-3.3-70B-Instruct"
SMALL_MODEL    = "meta-llama/Llama-3.2-8B-Instruct"   # RTX 3090, faster/cheaper


def train(
    model_id: str,
    data_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        raise RuntimeError(
            "Training dependencies not installed.\n"
            "Run:  pip install -r finetune/requirements_train.txt"
        )

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("Loading model in 4-bit (QLoRA) …")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=64,                      # rank — higher = more expressive, more VRAM
        lora_alpha=128,
        target_modules=[           # layers to adapt
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    log.info("Loading dataset: %s", data_path)
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_conversation(example: dict) -> dict:
        """Convert ShareGPT format to model chat template."""
        msgs = []
        for turn in example["conversations"]:
            role = {"system": "system", "human": "user", "gpt": "assistant"}.get(
                turn["from"], "user"
            )
            msgs.append({"role": role, "content": turn["value"]})
        return {"text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )}

    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
    log.info("Dataset: %d examples", len(dataset))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=512,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_cfg,
        args=training_args,
        tokenizer=tokenizer,
    )

    log.info("Starting fine-tune …")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Saved adapter → %s", output_dir)
    log.info("Next step:  python finetune/export_to_ollama.py --adapter %s", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune character LLM with LoRA")
    p.add_argument("--model",     default=DEFAULT_MODEL,
                   help=f"Base model (default: {DEFAULT_MODEL})")
    p.add_argument("--small",     action="store_true",
                   help="Use 8B model for RTX 3090 / lower VRAM")
    p.add_argument("--data",      default="finetune/data.jsonl")
    p.add_argument("--output",    default="finetune/adapter")
    p.add_argument("--epochs",    type=int,   default=3)
    p.add_argument("--batch",     type=int,   default=2)
    p.add_argument("--grad-accum",type=int,   default=4)
    p.add_argument("--lr",        type=float, default=2e-4)
    args = p.parse_args()

    model_id = SMALL_MODEL if args.small else args.model
    train(model_id, args.data, args.output, args.epochs, args.batch, args.grad_accum, args.lr)
