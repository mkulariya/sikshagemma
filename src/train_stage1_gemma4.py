import argparse
import json
import math
import os
import random
import statistics
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc

import unsloth
import torch
from datasets import Dataset, load_dataset
from transformers import AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt


DEFAULT_MODEL = "unsloth/gemma-4-E4B-it"
DEFAULT_DATASET_PATH = "sikshagemma_stage1_training_deduped.jsonl"
DEFAULT_OUTPUT_DIR = "stage1_gemma4_e4b"
DEFAULT_MAX_SEQ_LENGTH = 4096
RANDOM_SEED = 3407


class PreTokenizedCollator:
    """Pads pre‑tokenized lists to the longest in the batch, no tokenizer.pad."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids_padded != self.pad_token_id).long()
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Gemma-4 E4B – COLLATOR FIX")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--analyze-seq-len", action="store_true")
    parser.add_argument("--analysis-sample-size", type=int, default=5000)
    parser.add_argument("--save-adapter-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stage1_dataset(dataset_path, sample_size=0):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = standardize_sharegpt(dataset)
    if sample_size and sample_size > 0:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.select(range(sample_size))
    return dataset


def build_model_and_tokenizer(model_name, max_seq_length, load_in_4bit, lora_r, lora_alpha, seed):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=seed,
        use_gradient_checkpointing="unsloth",
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")
    return model, tokenizer


def build_tokenizer_only(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    text_tokenizer = get_text_tokenizer(processor)
    text_tokenizer = get_chat_template(text_tokenizer, chat_template="gemma-4-thinking")
    return text_tokenizer


def get_text_tokenizer(tokenizer):
    return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


def format_conversations(dataset, tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}
    return dataset.map(formatting_prompts_func, batched=True)


def pack_dataset(dataset, tokenizer, max_length, chunk_size=1000):
    text_tokenizer = get_text_tokenizer(tokenizer)
    all_texts = dataset["text"]

    lengths = []
    for start in range(0, len(all_texts), chunk_size):
        enc = text_tokenizer(
            all_texts[start : start + chunk_size],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        lengths.extend(len(ids) for ids in enc.input_ids)

    packed_texts = []
    current_text = ""
    current_len = 0
    for text, n in zip(all_texts, lengths):
        if current_len + n > max_length and current_text:
            packed_texts.append(current_text)
            current_text = text
            current_len = n
        else:
            current_text += text
            current_len += n
    if current_text:
        packed_texts.append(current_text)
    return Dataset.from_dict({"text": packed_texts})


def pre_tokenize_and_mask(
    packed_dataset,
    tokenizer,
    max_seq_length,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
):
    text_tokenizer = get_text_tokenizer(tokenizer)
    inst_ids = text_tokenizer.encode(instruction_part, add_special_tokens=False)
    resp_ids = text_tokenizer.encode(response_part, add_special_tokens=False)

    def _mask_and_tokenize(examples):
        input_ids_batch = []
        labels_batch = []
        for text in examples["text"]:
            enc = text_tokenizer(text, truncation=True, max_length=max_seq_length, padding=False, add_special_tokens=False)
            ids = enc.input_ids
            labels = [-100] * len(ids)

            i = 0
            while i <= len(ids) - len(resp_ids):
                if ids[i : i + len(resp_ids)] == resp_ids:
                    start = i
                    end = len(ids)
                    for j in range(start + len(resp_ids), len(ids) - len(inst_ids) + 1):
                        if ids[j : j + len(inst_ids)] == inst_ids:
                            end = j
                            break
                    for k in range(start, end):
                        labels[k] = ids[k]
                    i = end
                else:
                    i += 1

            input_ids_batch.append(ids)
            labels_batch.append(labels)

        return {"input_ids": input_ids_batch, "labels": labels_batch}

    tokenized = packed_dataset.map(
        _mask_and_tokenize,
        batched=True,
        remove_columns=packed_dataset.column_names,
        desc="Pre-tokenising + masking packed dataset",
        num_proc=2,
    )
    print(f"Pre‑tokenized dataset ready: {len(tokenized)} examples")
    if len(tokenized) > 0:
        sample_labels = tokenized[0]["labels"]
        num_trainable = sum(1 for x in sample_labels if x != -100)
        print(f"Sample 0: {len(sample_labels)} tokens, {num_trainable} trainable tokens")
    return tokenized


def percentile(sorted_values, p):
    if not sorted_values: return 0
    if len(sorted_values) == 1: return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lower, upper = math.floor(idx), math.ceil(idx)
    if lower == upper: return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (idx - lower)


def suggest_seq_len(p95, p99, max_seen):
    buckets = [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
    target = max(p99, p95)
    for b in buckets:
        if target <= b: return b
    return max_seen


def analyze_sequence_lengths(dataset, tokenizer, sample_size):
    if sample_size and sample_size > 0:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    formatted = format_conversations(dataset, tokenizer)
    text_tokenizer = get_text_tokenizer(tokenizer)
    lengths = [len(text_tokenizer(text, add_special_tokens=False).input_ids) for text in formatted["text"]]
    lengths.sort()
    if not lengths: raise ValueError("No token lengths computed.")
    stats = {
        "count": len(lengths), "min": lengths[0],
        "median": int(statistics.median(lengths)),
        "mean": round(sum(lengths) / len(lengths), 2),
        "p90": int(percentile(lengths, 0.90)),
        "p95": int(percentile(lengths, 0.95)),
        "p99": int(percentile(lengths, 0.99)),
        "max": lengths[-1],
    }
    stats["suggested_max_seq_length"] = suggest_seq_len(stats["p95"], stats["p99"], stats["max"])
    return stats


def latest_resume_checkpoint(output_dir, explicit_resume=None):
    if explicit_resume: return explicit_resume
    if not os.path.isdir(output_dir): return None
    return get_last_checkpoint(output_dir)


def print_gpu_stats():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    gpu_stats = torch.cuda.get_device_properties(0)
    start = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_mem} GB.")
    print(f"{start} GB of memory reserved.")
    return start, max_mem


def save_run_metadata(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train(args):
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    model, tokenizer = build_model_and_tokenizer(
        args.model_name, args.max_seq_length,
        args.load_in_4bit, args.lora_r, args.lora_alpha, args.seed
    )
    gc.collect()

    dataset = load_stage1_dataset(args.dataset_path, args.sample_size)
    formatted = format_conversations(dataset, tokenizer)
    del dataset; gc.collect()

    packed = pack_dataset(formatted, tokenizer, max_length=args.max_seq_length)
    del formatted; gc.collect()
    print(f"Packed {len(packed)} examples")

    pre_tokenized = pre_tokenize_and_mask(packed, tokenizer, args.max_seq_length)
    del packed; gc.collect()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=pre_tokenized,
        data_collator=None,   # we will set it manually after init
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field=None,
            packing=False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            #warmup_steps=args.warmup_steps,
            warmup_ratio=0.03,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            max_length=args.max_seq_length,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        ),
    )

    # Force the collator to be our safe one
    trainer.data_collator = PreTokenizedCollator(pad_token_id=tokenizer.pad_token_id)

    # torch.compile after trainer init
    # trainer.model = torch.compile(trainer.model, mode="reduce-overhead")
    
    if hasattr(trainer, "_unsloth_model_ref"):
        trainer._unsloth_model_ref = trainer.model

    torch.cuda.empty_cache()
    gc.collect()

    start_mem, max_mem = print_gpu_stats()
    resume_checkpoint = latest_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        print("No existing checkpoint found. Starting fresh.")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)

    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_for_lora = round(used_mem - start_mem, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_mem} GB.")
    print(f"Peak reserved memory for training = {used_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {round(used_mem/max_mem*100, 3)} %.")
    print(f"Peak reserved memory for training % of max memory = {round(used_for_lora/max_mem*100, 3)} %.")

    trainer.save_state()

    save_adapter_dir = args.save_adapter_dir or os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(save_adapter_dir)
    tokenizer.save_pretrained(save_adapter_dir)
    print(f"Saved final adapter to {save_adapter_dir}")

    save_run_metadata(
        os.path.join(args.output_dir, "stage1_run_summary.json"),
        {
            "dataset_path": args.dataset_path,
            "model_name": args.model_name,
            "output_dir": args.output_dir,
            "save_adapter_dir": save_adapter_dir,
            "max_seq_length": args.max_seq_length,
            "packed_examples": len(pre_tokenized),
            "train_runtime_seconds": trainer_stats.metrics.get("train_runtime"),
            "train_loss": trainer_stats.metrics.get("train_loss"),
            "resume_checkpoint": resume_checkpoint,
        },
    )


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.analyze_seq_len:
        dataset = load_stage1_dataset(args.dataset_path, sample_size=args.sample_size)
        tokenizer = build_tokenizer_only(args.model_name)
        stats = analyze_sequence_lengths(dataset, tokenizer, args.analysis_sample_size)
        print(json.dumps(stats, indent=2))
    else:
        train(args)