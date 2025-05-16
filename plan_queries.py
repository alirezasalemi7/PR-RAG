from vllm import LLM, SamplingParams
import argparse
from data.dataset import load_dataset
from data.formetters import get_query_planning_formatter
import json
import json5

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

def prepare_prompts(dataset, formater):
    reshaped_dataset = {
        "query": [],
        "id": [],
        "context": []
    }
    ids = []
    for data in dataset:
        reshaped_dataset["query"].append(data["query"])
        reshaped_dataset["id"].append(data["id"])
        ids.append(data["id"])
        if "context" in data:
            reshaped_dataset["context"].append(data["context"])
    return formater(reshaped_dataset), ids

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
parser.add_argument("--cache_dir", default="")

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_orig = load_dataset(args.inputs_addr, cache_dir = args.cache_dir)
    dataset = apply_num_generation(dataset_orig, args.num_generated_outputs)
    formater = get_query_planning_formatter()
    prompts, ids = prepare_prompts(dataset, formater)
    llm = LLM(model=args.model_addr)
    outputs_dict = {}
    temperature = args.temperature
    while prompts:
        sampling_params = SamplingParams(temperature=temperature, top_p=args.top_p, max_tokens=args.max_tokens, stop='```')
        outputs = llm.generate(prompts, sampling_params)
        failed_prompts = []
        failed_ids = []
        for id, prompt, output in zip(ids, prompts, outputs):
            if id not in outputs_dict:
                outputs_dict[id] = []
            json_text = output.outputs[0].text
            try:
                obj = parse_json(json_text)
            except Exception as e:
                failed_prompts.append(prompt)
                failed_ids.append(id)
                continue
            outputs_dict[id].append({
                "prompt": prompt,
                "parsed_queries": obj
            })
        prompts = failed_prompts
        ids = failed_ids
        if temperature < 1:
            temperature += 0.1
    with open(args.output_addr, "w") as file:
        json.dump(outputs_dict, file, indent=4)