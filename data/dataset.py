import datasets
import json

def load_dataset(addr, cache_dir):
    def gen():
        with open(addr, 'r') as f:
            try:
                questions = json.load(f)
                for q in questions:
                    if "retrieved_docs" in q:
                        yield {
                            "id": q["query_id"],
                            "query": q["query_description"],
                            "context": q["retrieved_docs"]
                        }
                    else:
                        yield {
                            "id": q["query_id"],
                            "query": q["query_description"]
                        }
            except:
                f.seek(0)
                for line in f:
                    if line.strip():
                        obj = json.loads(line.strip())
                        if "retrieved_docs" in obj:
                            yield {
                                "id": obj["query_id"],
                                "query": obj["query_description"],
                                "context": obj["retrieved_docs"]
                            }
                        else:
                            yield {
                                "id": obj["query_id"],
                                "query": obj["query_description"]
                            }
    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)

def load_dataset_with_plan(addr, cache_dir):
    def gen():
        with open(addr, 'r') as f:
            questions = json.load(f)
            for q in questions:
                plan = []
                for a in q["plans"][0]['parsed_queries']:
                    if 'aspect' not in a or 'query' not in a or 'reason' not in a:
                        a['reason'] = a['aspect']
                    plan.append({
                        "aspect": a["aspect"],
                        "query": a["query"],
                        "reason": a["reason"],
                        "context": a["retrieved_docs"]
                    })
                yield {
                    "id": q["query_id"],
                    "query": q["query_description"],
                    "plan": plan
                }
    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)

def load_dataset_with_plan_for_train(addr, cache_dir):
    def gen():
        with open(addr, 'r') as f:
            questions = json.load(f)
            for q in questions:
                for i, p in enumerate(q["plans"]):
                    plan = []
                    for a in p['parsed_queries']:
                        if 'aspect' not in a or 'query' not in a or 'reason' not in a:
                            print(f"Error in query: {a}")
                            continue
                        plan.append({
                            "aspect": a["aspect"],
                            "query": a["query"],
                            "reason": a["reason"],
                            "context": a["retrieved_docs"]
                        })
                    yield {
                        "id": f'{q["query_id"]}@{i}',
                        "query": q["query_description"],
                        "plan": plan
                    }
    return datasets.Dataset.from_generator(gen, cache_dir=cache_dir)

def load_responses(addr):
    outputs = dict()
    with open(addr, 'r') as f:
        temp = json.load(f)
    for k, v in temp.items():
        for i, o in enumerate(v):
            outputs[f'{k}@{i}'] = o
    return outputs

def load_responses_global_local_search(addr, addr_init, max_global=-1, max_local=-1):
    outputs_grouped, responses_indv = dict(), dict()
    if addr:
        with open(addr, 'r') as f:
            temp = json.load(f)
    else:
        temp = dict()
    with open(addr_init, 'r') as f:
        temp_init = json.load(f)
        temp['-1'] = temp_init
    for local_step, local_step_outputs in temp.items():
        if max_local > -1 and int(local_step) >= max_local:
            break
        for k, v in local_step_outputs.items():
            if k not in outputs_grouped:
                outputs_grouped[k] = []
            for i, o in enumerate(v):
                if max_global > -1 and i >= max_global:
                    break
                o['new_id'] = f'{k}@{local_step}@{i}'
                outputs_grouped[k].append(o)
                responses_indv[f'{k}@{local_step}@{i}'] = o
    return outputs_grouped, responses_indv