import json
import torch
import argparse

import numpy as np

from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from icat.retriever import Retriever
from icat.llm_eval import LLMEvaluator
import logging


class CoverageScore:
    def __init__(self, 
                 nli_model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 corpus_path: str = "",
                 nli_batch_size: int = 1,
                 llm_batch_size: int = 1024, train=False,
                 cache_dir: Optional[str] = ""):
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.train = train
        
        self.logger.info("Initializing CoverageScore...")
        self.retriever = Retriever(cache_dir=cache_dir)
        self.topk = 10
        self.retriever.process_corpus(corpus_path)
        self.llm_evaluator = LLMEvaluator()
        
        # Initialize NLI model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        self.nli_model.eval()

        self.nli_batch_size = nli_batch_size
        self.llm_batch_size = llm_batch_size

    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        inputs = self.nli_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.nli_model(**inputs)
        
        prediction = torch.softmax(output.logits[0], -1).cpu().numpy()
        return bool(prediction[0] > 0.5)  # Index 0 corresponds to entailment

    def _check_entailment_batch(self, premises: List[str], hypotheses: List[str]) -> List[bool]:
        # Process multiple premise-hypothesis pairs at once
        if self.train:
            inputs = self.nli_tokenizer(
                premises, 
                hypotheses, 
                truncation=True, 
                padding=True, 
                return_tensors="pt",
                max_length=400
            )
        else:
            inputs = self.nli_tokenizer(
                premises, 
                hypotheses, 
                truncation=True, 
                padding=True, 
                return_tensors="pt",
                max_length=512
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.nli_model(**inputs)
        
        predictions = torch.softmax(output.logits, -1).cpu().numpy()
        results = [bool(pred[0] > 0.5) for pred in predictions]

        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        output = output.logits.to("cpu")
        torch.cuda.empty_cache()
        return results

    def coverage_score(self, queries: List[str], model_responses: List[str]) -> List[Dict]:
        self.logger.info(f"Processing {len(queries)} queries")
        
        assert len(queries) == len(model_responses), "Number of queries and responses must match"
        
        # Generate topics using LLM for all queries
        self.logger.info("Generating topics for queries...")
        topics_prompts = [
            f'given this query "{query}" generate all the possible subtopics or related queries from most important to least, up to 10, one in each line with this jsonl format {{"topic": ...}}, nothing else in your response'
            for query in queries
        ]
        generated_topics_raw = self.llm_evaluator.generate(topics_prompts)

        print("generated topics raw:")
        print(generated_topics_raw)
        
        # Process generated topics for each query
        self.logger.info("Parsing generated topics...")
        all_generated_topics = []
        for query_idx, topics_raw in enumerate(generated_topics_raw):
            generated_topics = []
            for line in topics_raw.split('\n'):
                if line.strip().startswith('{"topic":'):
                    try:
                        topic = json.loads(line.strip())["topic"]
                        generated_topics.append(topic)
                    except json.JSONDecodeError:
                        continue
            all_generated_topics.append(generated_topics)
            self.logger.info(f"Query {query_idx + 1}: Generated {len(generated_topics)} topics")
        
        print("all generated topics:")
        print(all_generated_topics)

        # Get atomic facts for all responses
        self.logger.info("Generating atomic facts from responses...")
        all_facts = self.llm_evaluator.generate_facts(model_responses)
        
        # Process entailment for each query's facts in batches
        all_entailed_facts = []
        for query_idx, (query, response_facts) in enumerate(zip(queries, all_facts)):
            self.logger.info(f"Processing entailment for query {query_idx + 1}")
            entailed_facts = []
            
            # Batch process facts for entailment
            for i in range(0, len(response_facts), self.nli_batch_size):
                batch_facts = response_facts[i:i + self.nli_batch_size]
                batch_results = []
                
                for fact in batch_facts:
                    top_docs = self.retriever.retrieve(fact, top_k=self.topk)
                    premises = [doc[1] for doc in top_docs]  # Get snippets
                    hypotheses = [fact] * len(premises)  # Repeat fact for each premise
                    
                    # Check entailment for current fact against all its premises
                    entailment_results = self._check_entailment_batch(premises, hypotheses)
                    
                    if any(entailment_results):
                        # Store the first document that entails this fact
                        for is_entailed, doc in zip(entailment_results, top_docs):
                            if is_entailed:
                                batch_results.append((doc[0], len(entailed_facts), fact))
                                break
                
                entailed_facts.extend(batch_results)
            
            all_entailed_facts.append(entailed_facts)

        # Collect all coverage prompts
        coverage_prompts = []
        for query_idx, (query, response_facts) in enumerate(zip(queries, all_entailed_facts)):
            # Format entailed facts with numbers for this query
            entailed_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate([f[2] for f in response_facts])])
            
            coverage_prompt = (
                f'given this query "{query}", the following list of subtopics:\n\n' +
                "\n".join([f"{j+1} : {topic}" for j, topic in enumerate(all_generated_topics[query_idx])]) + "\n\n" +
                f'return the subtopics that are covered in the given text below with a list of facts, '
                f'mention each subtopic only once with a list of fact numbers for each subtopic, '
                f'the fact numbers should reference the most relevant facts that support the subtopic, '
                f'they should be explicitly mentioned in the given text, if they are not explicitly mentioned '
                f'don\'t include them in your response, if some subtopic is not covered without any evidence '
                f'don\'t include it in your response, use this jsonl format '
                f'{{"topic_id": ..., "evidence": [fact_number, ...]}}, one json object per line, '
                f'here is the text with enumerated facts:\n\n{entailed_text}'
            )
            coverage_prompts.append(coverage_prompt)

        # Process coverage prompts in batches
        covered_topics_responses = []
        for i in range(0, len(coverage_prompts), self.llm_batch_size):
            batch_prompts = coverage_prompts[i:i + self.llm_batch_size]
            batch_responses = self.llm_evaluator.generate(batch_prompts)
            covered_topics_responses.extend(batch_responses)

        results = []
        # Process each query-response pair
        for query_idx, (query, response_facts, covered_topics_raw) in enumerate(zip(queries, all_entailed_facts, covered_topics_responses)):
            total_topics = len(all_generated_topics[query_idx])
            covered_data = []
            seen_topic_ids = set()
            
            # Get the correct atomic facts for this query
            query_atomic_facts = all_facts[query_idx]
            total_facts = len(query_atomic_facts)
            
            # Parse coverage response
            for line in covered_topics_raw.split('\n'):
                if line.strip().startswith('{"topic_id":'):
                    try:
                        data = json.loads(line.strip())
                        topic_id = int(data["topic_id"]) - 1
                        
                        # Skip if evidence list is empty
                        if not data["evidence"]:
                            continue
                        
                        if (0 <= topic_id < total_topics) and (topic_id not in seen_topic_ids):
                            valid_evidence = []
                            for fact_num in data["evidence"]:
                                fact_idx = int(fact_num) - 1
                                if 0 <= fact_idx < total_facts:
                                    valid_evidence.append(fact_idx)
                            
                            if valid_evidence:
                                seen_topic_ids.add(topic_id)
                                covered_data.append({
                                    "topic_id": topic_id + 1,
                                    "evidence": valid_evidence
                                })
                    except:
                        continue

            coverage_score = len(covered_data) / total_topics if total_topics > 0 else 0
            precision = len(response_facts) / total_facts if total_facts > 0 else 0
            f1 = 2 * (precision * coverage_score) / (precision + coverage_score) if (precision + coverage_score) > 0 else 0

            self.logger.info(f"Query {query_idx + 1} metrics - Coverage: {coverage_score:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}")

            results.append({
                "generated_topics": all_generated_topics[query_idx],
                "atomic_facts": query_atomic_facts,
                "entailed_facts": [f[2] for f in response_facts],
                "covered_topics": covered_data,
                "metrics": {
                    "coverage_score": coverage_score,
                    "precision": precision,
                    "f1": f1
                }
            })
        
        return results

def get_response(response, cot):
    if not cot:
        return response['output']
    else:
        if "response:" in response['output']:
            return response['output'].split("response:")[1].strip()
        else:
            return response['output']

parser = argparse.ArgumentParser()
parser.add_argument('--corpus-path', required=True)
parser.add_argument("--cache_dir", default="")
parser.add_argument("--queries", type=str, required=True)
parser.add_argument("--responses", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--aggregate", action="store_true")
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--train", action="store_true", help="used for training")
parser.add_argument("--local_search", action="store_true", help="evaluating local search")
parser.add_argument("--cot", action="store_true")
    

if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.queries, "r") as f:
        queries = dict()
        queries_list = []
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                queries_list.append(data)        
        shard_size = (len(queries_list) + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, len(queries_list))
        queries_list = queries_list[start_idx:end_idx]
        for data in queries_list:
            queries[data["query_id"]] = data["query_description"]
    
    with open(args.responses, "r") as f:
        all_responses = json.load(f)
    
    if args.local_search:
        all_results = {}
        coverage_scorer = CoverageScore(corpus_path=args.corpus_path, train=args.train, cache_dir=args.cache_dir)
        for key, responses in all_responses.items():
            processed_queries = []
            query_ids = []
            processed_responses = []
            for query_id in queries.keys():
                for response in responses[query_id]:
                    query_ids.append(query_id)
                    processed_queries.append(queries[query_id])
                    processed_responses.append(get_response(response, args.cot))

            results = coverage_scorer.coverage_score(
                queries=processed_queries,
                model_responses=processed_responses
            )
            final_results = dict()
            for query_id, result in zip(query_ids, results):
                if query_id not in final_results:
                    final_results[query_id] = []
                final_results[query_id].append(result)
            
            if args.aggregate:
                average = {
                    "coverage_score": 0,
                    "precision": 0,
                    "f1": 0
                }
                for query_id, result in final_results.items():
                    average["coverage_score"] += result[0]["metrics"]["coverage_score"] / len(final_results)
                    average["precision"] += result[0]["metrics"]["precision"] / len(final_results)
                    average["f1"] += result[0]["metrics"]["f1"] / len(final_results)
            all_results[key] = {
                "average": average if args.aggregate else None,
                "per_query": final_results,
            }
        with open(args.output + f"_{args.shard_id}", "w") as f:
            json.dump(all_results, f, indent=4)
    else:
        responses = all_responses
        processed_queries = []
        query_ids = []
        processed_responses = []
        for query_id in queries.keys():
            for response in responses[query_id]:
                query_ids.append(query_id)
                processed_queries.append(queries[query_id])
                processed_responses.append(get_response(response, args.cot))

        coverage_scorer = CoverageScore(corpus_path=args.corpus_path, train=args.train, cache_dir=args.cache_dir)
        results = coverage_scorer.coverage_score(
            queries=processed_queries,
            model_responses=processed_responses
        )
        final_results = dict()
        for query_id, result in zip(query_ids, results):
            if query_id not in final_results:
                final_results[query_id] = []
            final_results[query_id].append(result)
        
        if args.aggregate:
            average = {
                "coverage_score": 0,
                "precision": 0,
                "f1": 0
            }
            for query_id, result in final_results.items():
                average["coverage_score"] += result[0]["metrics"]["coverage_score"] / len(final_results)
                average["precision"] += result[0]["metrics"]["precision"] / len(final_results)
                average["f1"] += result[0]["metrics"]["f1"] / len(final_results)
        
        with open(args.output + f"_{args.shard_id}", "w") as f:
            json.dump({
                "average": average if args.aggregate else None,
                "per_query": final_results,
            }, f, indent=4)