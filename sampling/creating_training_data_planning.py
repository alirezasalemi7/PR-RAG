import json
import argparse


def top_k_percent_threshold(values, k):
    sorted_values = sorted(values)
    threshold_index = max(0, int(len(sorted_values) * k) - 1)
    return sorted_values[threshold_index]

parser = argparse.ArgumentParser()
parser.add_argument("--score_file", required=True)
parser.add_argument("--input_plans", required=True)
parser.add_argument("--training_data", required=True)
parser.add_argument("--percentile", type=int, default=1)

if __name__ == "__main__":

    args = parser.parse_args()

    input_data = args.input_plans
    output_data = args.training_data
    score_files = args.score_file
    method = "percentile"
    percentile = args.percentile

    with open(input_data, "r") as f:
        dataset = json.load(f)
    with open(score_files, "r") as f:
        scores = json.load(f)

    new_dataset = []
    

    for data in dataset:
        qid = data['query_id']
        q_scores = [score['metrics']['f1'] for score in scores[qid]]
        if method == "percentile":
            threshold = top_k_percent_threshold(q_scores, percentile)
        elif method == "max":
            threshold = max(q_scores) - 1e-7

        for plan, score in zip(data['plans'], q_scores):
            for parsed in plan['parsed_queries']:
                del parsed['retrieved_docs']
            if score > threshold:
                new_dataset.append({
                    "query_id": qid,
                    "query_description": data['query_description'],
                    "output": json.dumps(plan['parsed_queries'], indent=1),
                    "output_score": score
                })

    print(len(new_dataset))
    with open(output_data, "w") as f:
        json.dump(new_dataset, f, indent=4)