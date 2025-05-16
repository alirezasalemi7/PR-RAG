from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import datasets
import random
import glob
import os
import json


@dataclass
class DataArguments():
    train_data_address : str = field(
        default = 'train.jsonl',
        metadata={'help': 'where to get data'}
    )

    model_address : str = field(
        default = 'model',
        metadata={'help': 'where the model is'}
    )

    cache_dir : str = field(
        default = './cache/',
        metadata={'help': 'where to store things'}
    )

    max_len_input : int = field(
        default = 128,
        metadata={'help': 'maximum length input'}
    )

def get_collator(max_length, tokenizer):
    def collator(batch):
        pos_outputs = [x['pos_output'] for x in batch]
        neg_outputs = [x['neg_output'] for x in batch]
        pos_encoded = tokenizer(
            pos_outputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        neg_encoded = tokenizer(
            neg_outputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pos_input_ids": pos_encoded["input_ids"],
            "pos_attention_mask": pos_encoded["attention_mask"],
            "neg_input_ids": neg_encoded["input_ids"],
            "neg_attention_mask": neg_encoded["attention_mask"]
        }
    return collator

class RewardTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.reward_loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        pos_input_ids = inputs["pos_input_ids"]
        pos_attention_mask = inputs["pos_attention_mask"]
        neg_input_ids = inputs["neg_input_ids"]
        neg_attention_mask = inputs["neg_attention_mask"]

        pos_outputs = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
        neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
        pos_scores = pos_outputs.logits.squeeze(-1)
        neg_scores = neg_outputs.logits.squeeze(-1)

        score_diff = pos_scores - neg_scores
        labels = torch.ones(score_diff.size(), device=pos_scores.device)
        reward_loss = self.reward_loss_fn(score_diff, labels)
        labels.to(torch.device("cpu"))

        return (reward_loss, pos_outputs) if return_outputs else reward_loss

def create_processor(max_len):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)
    return preprocess_function

if __name__ == "__main__":
    
    parser = HfArgumentParser([DataArguments, TrainingArguments])
    data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = AutoModelForSequenceClassification.from_pretrained(data_args.model_address, num_labels=1, cache_dir=data_args.cache_dir, reference_compile=False)
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_address, cache_dir=data_args.cache_dir)
    
    training_args.remove_unused_columns=False
    
    processor = create_processor(data_args.max_len_input)
    
    with open(data_args.train_data_address, "r") as f:
        temp_dataset = json.load(f)
        dataset_list = []
        for data in temp_dataset:
            dataset_list.append({
                "pos_output": f'{data["query_description"]}{tokenizer.sep_token}{data['output_pos']}',
                "neg_output": f'{data["query_description"]}{tokenizer.sep_token}{data['output_neg']}'
            })
    train_dataset = datasets.Dataset.from_list(dataset_list)
    collator = get_collator(data_args.max_len_input, tokenizer)
    trainer = RewardTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = train_dataset,
        tokenizer = tokenizer
    )
    trainer.train()