import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL = "merged_stage1"
DEFAULT_ADAPTER_DIR = "stage2_gemma4_e4b/final_adapter"
DEFAULT_OUTPUT_DIR = "merged_stage2"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Stage 2 LoRA adapter into base model (bf16)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError(f"Adapter dir not found: {args.adapter_dir}")

    print(f"Loading base model in bf16: {args.model_name}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading adapter: {args.adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging adapter into base weights...")
    merged = peft_model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged model to: {args.output_dir}")
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(args.model_name).save_pretrained(args.output_dir)

    print("Done. Merged model ready for Stage 3 training.")
    print("Files:", os.listdir(args.output_dir))


if __name__ == "__main__":
    main()
