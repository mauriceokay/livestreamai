"""
Merges the LoRA adapter into the base model and exports to GGUF format
so it can be loaded by Ollama — which brain.py already supports.

Run:
  python finetune/export_to_ollama.py --adapter finetune/adapter

What it does:
  1. Merges LoRA weights back into the base model (full precision)
  2. Converts merged model to GGUF (Q5_K_M quantization — good quality/size)
  3. Creates an Ollama Modelfile and registers the model as "maya"
  4. You can then set LLM_BACKEND=ollama + OLLAMA_MODEL=maya in .env

Requirements:
  pip install -r finetune/requirements_train.txt
  git clone https://github.com/ggerganov/llama.cpp   (for conversion script)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LLAMA_CPP    = PROJECT_ROOT / "llama.cpp"


def export(adapter_dir: str, model_name: str, quant: str) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError(
            "Run:  pip install -r finetune/requirements_train.txt"
        )

    adapter_path = Path(adapter_dir)
    merged_path  = adapter_path / "merged"
    gguf_path    = PROJECT_ROOT / "models" / f"{model_name}.gguf"
    gguf_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Merge adapter into base model ────────────────────────────
    print("Step 1/3 — Merging LoRA adapter into base model …")
    config_path = adapter_path / "adapter_config.json"
    import json
    with config_path.open() as f:
        base_model_id = json.load(f)["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",          # merge on CPU to avoid OOM
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_path))
    print(f"  Merged model saved → {merged_path}")

    # ── Step 2: Convert to GGUF ───────────────────────────────────────────
    print("Step 2/3 — Converting to GGUF …")
    if not LLAMA_CPP.exists():
        print(
            "\nllama.cpp not found. Clone it first:\n"
            f"  git clone https://github.com/ggerganov/llama.cpp {LLAMA_CPP}\n"
            "  cd llama.cpp && pip install -r requirements.txt"
        )
        sys.exit(1)

    convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"
    gguf_f16 = gguf_path.with_suffix(".f16.gguf")
    subprocess.run([
        sys.executable, str(convert_script),
        str(merged_path),
        "--outfile", str(gguf_f16),
        "--outtype", "f16",
    ], check=True)

    # Quantise to Q5_K_M (good quality, ~half the size of f16)
    quantize_bin = LLAMA_CPP / "llama-quantize"
    if quantize_bin.exists():
        subprocess.run([
            str(quantize_bin),
            str(gguf_f16),
            str(gguf_path),
            quant,
        ], check=True)
        gguf_f16.unlink()
        print(f"  GGUF saved → {gguf_path}  ({quant})")
    else:
        gguf_path = gguf_f16
        print(
            f"  llama-quantize not built — using f16 GGUF → {gguf_path}\n"
            "  (build llama.cpp with: cd llama.cpp && cmake -B build && cmake --build build -j)"
        )

    # ── Step 3: Register with Ollama ──────────────────────────────────────
    print("Step 3/3 — Registering with Ollama …")
    modelfile_path = gguf_path.parent / f"{model_name}.Modelfile"
    modelfile_path.write_text(
        f'FROM {gguf_path.resolve()}\n'
        f'PARAMETER temperature 0.85\n'
        f'PARAMETER num_predict 120\n'
        f'PARAMETER stop "<|eot_id|>"\n'
    )

    if shutil.which("ollama"):
        subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)], check=True)
        print(f"\n  Model registered as '{model_name}' in Ollama.")
        print(f"\n  Add to your .env:")
        print(f"    LLM_BACKEND=ollama")
        print(f"    OLLAMA_MODEL={model_name}")
    else:
        print(
            "\nOllama not found. Install from https://ollama.ai, then run:\n"
            f"  ollama create {model_name} -f {modelfile_path}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="finetune/adapter")
    p.add_argument("--name",    default="maya",
                   help="Model name in Ollama (default: maya)")
    p.add_argument("--quant",   default="Q5_K_M",
                   help="GGUF quantization level (default: Q5_K_M)")
    args = p.parse_args()
    export(args.adapter, args.name, args.quant)
