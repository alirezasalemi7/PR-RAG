import json
import random
import argparse

def top_k_percent_threshold(values, k):
    sorted_values = sorted(values)
    threshold_index = max(0, int(len(sorted_values) * k) - 1)
    return sorted_values[threshold_index]

parser = argparse.ArgumentParser()
parser.add_argument("--score_file", required=True)
parser.add_argument("--input_plans", required=True)
parser.add_argument("--training_data", required=True)
parser.add_argument("--response_file", required=True)
parser.add_argument("--threshold", type=int, default=0.1)
parser.add_argument("--num_samples", type=int, default=8)

if __name__ == "__main__":
    args = parser.parse_args()
    input_data = args.input_plans
    output_data = args.training_data
    respose_data = args.response_file
    score_files = args.score_file

    max_num_samples = args.num_samples
    positive_ratio = 0.05
    min_diff = args.threshold

    with open(input_data, "r") as f:
        dataset = json.load(f)
    with open(score_files, "r") as f:
        scores = json.load(f)
    with open(respose_data, "r") as f:
        responses = json.load(f)

    new_dataset = []

    for data in dataset:
        qid = data['query_id']
        q_scores = [score['metrics']['f1'] for score in scores[qid]]
        q_responses = [response['output'] for response in responses[qid]]
        q_plans = [plan['parsed_queries'] for plan in data['plans']]
        for plan in q_plans:
            for parsed in plan:
                del parsed['retrieved_docs']
        positive_outputs = []
        negative_outputs = []
        max_score = max(q_scores)
        for score, response, plan in zip(q_scores, q_responses, q_plans):
            if score > max_score * (1 - positive_ratio):
                positive_outputs.append((response, score, plan))
            else:
                negative_outputs.append((response, score, plan))
        if len(positive_outputs) == 0 or len(negative_outputs) == 0:
            continue
        for _ in range(max_num_samples):
            score_neg = 0
            score_pos = 0
            num_try = 0
            failed = False
            while score_pos - score_neg < min_diff:
                num_try += 1
                response_pos, score_pos, plan_pos = random.choice(positive_outputs)
                response_neg, score_neg, plan_neg = random.choice(negative_outputs)
                if num_try > 100:
                    failed = True
                    break
            if failed:
                continue
            new_dataset.append({
                "query_id": qid,
                "query_description": data['query_description'],
                "output_pos": response_pos,
                "output_neg": response_neg,
                "plan_pos": plan_pos,
                "plan_neg": plan_neg,
                "output_pos_score": score_pos,
                "output_neg_score": score_neg,
            })

    print(len(new_dataset))
    with open(output_data, "w") as f:
        json.dump(new_dataset, f, indent=4)