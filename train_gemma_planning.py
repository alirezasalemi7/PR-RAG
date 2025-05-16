from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
from data.formetters import get_query_planning_formatter
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import datasets

parser = argparse.ArgumentParser()

parser.add_argument("--data_addr", required=True)
parser.add_argument("--cache_dir", default="")
parser.add_argument("--model_addr", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--per_device_train_batch_size", type=int, default=64)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--max_seq_length", type=int, default=4096)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.data_addr, "r") as f:
        temp_dataset = json.load(f)
        dataset = datasets.Dataset.from_list(
            [{
                "query": data["query_description"],
                "output": data["output"]
            } for data in temp_dataset]
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_addr, cache_dir = args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_addr, cache_dir = args.cache_dir)
    formatter = get_query_planning_formatter(True)
    collator = DataCollatorForCompletionOnlyLM(response_template="output: ```json", tokenizer=tokenizer)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        save_only_model=True
    )
    trianer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=formatter,
        data_collator=collator
    )

    trianer.train()
