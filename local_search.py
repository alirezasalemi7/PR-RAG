from vllm import LLM, SamplingParams
import argparse
from data.dataset import load_dataset_with_plan_for_train, load_responses
from data.formetters import get_query_planning_local_search_formatter
from transformers import AutoTokenizer
import json


def prepare_prompts(dataset, initial_responses, formater):
    reshaped_dataset = {
        "query": [],
        "id": [],
        "plan": [],
        "output_neg": [],
        "output_pos": []
    }
    for data in dataset:
        reshaped_dataset["query"].append(data["query"])
        reshaped_dataset["id"].append(data["id"])
        output_neg = initial_responses[data["id"]]["output"]
        output_pos = ""
        reshaped_dataset["output_neg"].append(output_neg)
        reshaped_dataset["output_pos"].append(output_pos)
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
parser.add_argument("--initial_response_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--num_iterations", type=int, default=1)
parser.add_argument("--cache_dir", default="")

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_addr, cache_dir = args.cache_dir)
    dataset_orig = load_dataset_with_plan_for_train(args.inputs_addr, cache_dir = args.cache_dir)
    dataset = apply_num_generation(dataset_orig, 1)
    initial_responses = load_responses(args.initial_response_addr)
    formater = get_query_planning_local_search_formatter(False)
    all_outputs_dict = {}
    llm = LLM(model=args.model_addr)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, logprobs=1)
    for i in range(args.num_iterations):
        prompts, ids = prepare_prompts(dataset, initial_responses, formater)
        outputs = llm.generate(prompts, sampling_params)
        outputs_dict = {}
        next_round_dict = {}
        for id, prompt, output in zip(ids, prompts,  outputs):
            next_round_dict[id] = {
                "prompt": prompt,
                "output": output.outputs[0].text,
                "log_prob": output.outputs[0].cumulative_logprob
            }
            orig_id, index = id.split("@")
            if orig_id not in outputs_dict:
                outputs_dict[orig_id] = []
            
            outputs_dict[orig_id].append({
                "prompt": prompt,
                "output": output.outputs[0].text,
                "log_prob": output.outputs[0].cumulative_logprob
            })
        all_outputs_dict[i] = outputs_dict
        initial_responses = next_round_dict
    with open(args.output_addr, "w") as file:
        json.dump(all_outputs_dict, file, indent=4)