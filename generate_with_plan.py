from vllm import LLM, SamplingParams
import argparse
from data.dataset import load_dataset_with_plan, load_dataset_with_plan_for_train
from data.formetters import get_query_planning_rag_formatter
from transformers import AutoTokenizer
import json


def prepare_prompts(dataset, formater):
    reshaped_dataset = {
        "query": [],
        "id": [],
        "plan": []
    }
    for data in dataset:
        reshaped_dataset["query"].append(data["query"])
        reshaped_dataset["id"].append(data["id"])
        if "plan" in data:
            reshaped_dataset["plan"].append(data["plan"])
    return formater(reshaped_dataset), reshaped_dataset["id"]

def apply_num_generation(dataset, num_generation):
    new_dataset = []
    for data in dataset:
        for i in range(num_generation):
            new_dataset.append(data)
    return new_dataset

def post_process_output_based_on_num_generation(output, num_generation):
    new_output = []
    temp = []
    for out in output:
        temp.append(out)
        if len(temp) == num_generation:
            new_output.append(temp)
            temp = []
    return new_output

parser = argparse.ArgumentParser()
parser.add_argument("--model_addr", type=str, required=True)
parser.add_argument("--inputs_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--num_generated_outputs", type=int, default=1)
parser.add_argument("--num_contexts", type=int, default=5)
parser.add_argument("--for_train_or_global_search", action="store_true")
parser.add_argument("--cache_dir", default="")

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_addr, cache_dir = args.cache_dir)
    if args.for_train_or_global_search:
        dataset_orig = load_dataset_with_plan_for_train(args.inputs_addr, cache_dir = args.cache_dir)
    else:
        dataset_orig = load_dataset_with_plan(args.inputs_addr, cache_dir = args.cache_dir)
    dataset = apply_num_generation(dataset_orig, args.num_generated_outputs)
    formater = get_query_planning_rag_formatter(args.num_contexts)
    prompts, ids = prepare_prompts(dataset, formater)
    llm = LLM(model=args.model_addr)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, logprobs=1)
    outputs = llm.generate(prompts, sampling_params)
    outputs_dict = {}
    for id, prompt, output in zip(ids, prompts,  outputs):
        if args.for_train_or_global_search:
            orig_id, index = id.split("@")
            if orig_id not in outputs_dict:
                outputs_dict[orig_id] = []
            
            outputs_dict[orig_id].append({
                "prompt": prompt,
                "output": output.outputs[0].text,
                "log_prob": output.outputs[0].cumulative_logprob
            })
        else:
            if id not in outputs_dict:
                outputs_dict[id] = []
            outputs_dict[id].append(
                {
                    "prompt": prompt,
                    "output": output.outputs[0].text,
                    "log_prob": output.outputs[0].cumulative_logprob
                }
            )
    with open(args.output_addr, "w") as file:
        json.dump(outputs_dict, file, indent=4)