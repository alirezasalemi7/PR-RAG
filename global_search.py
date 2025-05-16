import argparse
from data.dataset import load_dataset, load_responses_global_local_search
from data.formetters import get_query_planning_global_search_formatter
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import json5
from datasets import Dataset

def pair_items_two_by_two(items):
    pairs = [(items[i], items[i+1]) for i in range(0, len(items) - 1, 2)]
    if len(items) % 2 != 0:
        pairs.append((items[-1],))  # Add the last single item as a tuple
    return pairs

def apply_num_generation(dataset, num_generation):
    new_dataset = []
    for data in dataset:
        for i in range(num_generation):
            new_dataset.append(data)
    return new_dataset

def parse_json(json_str):
    json_str = json_str.replace("```json", "").replace("```", "")
    try:
        return json.loads(json_str, strict=False)
    except Exception as e:
        pass
    try:
        return json5.loads(json_str)
    except Exception as e:
        raise Exception("Could not parse the input")

def prepare_prompts(dataset, responses, sep_token):
    inputs = []
    ids_q = []
    ids_o = []
    for data in dataset:
        id = data["id"]
        q_responses = responses[id]
        for response in q_responses:
            inputs.append({"text": f'{data["query"]} {sep_token} {response["output"]}'})
            ids_q.append(id)
            ids_o.append(response["new_id"])
    return Dataset.from_list(inputs), ids_q, ids_o

def gen_proccessor(tokenizer, max_input_length):
    def process(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_input_length, padding="max_length")
    return process

parser = argparse.ArgumentParser()
parser.add_argument("--model_addr", type=str, required=True)
parser.add_argument("--inputs_addr", type=str, required=True)
parser.add_argument("--responses_addr", type=str, required=False, default="")
parser.add_argument("--init_responses_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--max_local", type=int, default=-1)
parser.add_argument("--max_global", type=int, default=-1)
parser.add_argument("--cache_dir", default="")

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_orig = load_dataset(args.inputs_addr, cache_dir = args.cache_dir)
    responses_grouped, responses_indv = load_responses_global_local_search(args.responses_addr, args.init_responses_addr, max_global=args.max_global, max_local=args.max_local)
    dataset = apply_num_generation(dataset_orig, 1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_addr, cache_dir = args.cache_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_addr, cache_dir = args.cache_dir, reference_compile=False)
    except:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_addr, cache_dir = args.cache_dir)
    inputs, ids_question, ids_output = prepare_prompts(dataset, responses_grouped, tokenizer.sep_token)
    inputs = inputs.map(gen_proccessor(tokenizer, args.max_tokens), batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_tokens, pad_to_multiple_of=512)
    arguments = TrainingArguments(
        output_dir="temp",
        per_device_eval_batch_size=32,
        eval_accumulation_steps=1,
        do_predict=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        data_collator=collator,
        tokenizer=tokenizer
    )
    all_grouped_scores = {}
    scores = trainer.predict(inputs).predictions.squeeze().tolist()
    for id_q, id_o, score in zip(ids_question, ids_output, scores):
        if id_q not in all_grouped_scores:
            all_grouped_scores[id_q] = []
        all_grouped_scores[id_q].append({"new_id": id_o, "score": score})
    final_outputs = {}
    for id_q, scores in all_grouped_scores.items():
        all_grouped_scores[id_q] = sorted(scores, key=lambda x: x["score"], reverse=True)
        final_outputs[id_q] = [responses_indv[all_grouped_scores[id_q][0]["new_id"]]]
    with open(args.output_addr, 'w') as f:
        json.dump(final_outputs, f, indent=4)
        