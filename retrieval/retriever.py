import json
import pickle
import faiss
import nltk
import numpy as np
import math

from typing import List, Dict, Tuple
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import argparse


class Retriever:
    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-l", cache_dir: str = "", batch_size: int = 512):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index = None
        self.snippets = []
        self.doc_ids = []
        nltk.download('punkt')
        
    def create_snippets(self, text: str, max_length: int = 512, stride: int = 96) -> List[str]:
        """Create overlapping snippets of up to max_length words with stride stride."""
        # Split on whitespace and filter out empty strings
        words = [w for w in text.split() if w]
        return [
            " ".join(words[i:min(i + max_length, len(words))])
            for i in range(0, len(words), stride)
        ]
    
    def process_corpus(self, corpus_path: str):
        """Process corpus and create FAISS index."""
        print(f"Processing corpus from: {corpus_path}")
        snippets_cache_file = self.cache_dir / "snippets_embeddings_large.pkl"
        index_cache_file = self.cache_dir / "faiss_index_large.bin"

        # If no cache exists, process as before
        if snippets_cache_file.exists():
            print("Loading cached embeddings...")
            with open(snippets_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.snippets = cached_data['snippets']
                self.doc_ids = cached_data['doc_ids']
                embeddings = cached_data['embeddings'].astype(np.float16)
            print(f"Loaded {len(self.snippets)} snippets from cache")
        else:
            print("Cache not found. Processing corpus...")
            # Process corpus
            all_snippets = []
            all_doc_ids = []
            
            with open(corpus_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    if i % 1000 == 0:  # Log progress every 1000 documents
                        print(f"Processed documents: {i}...")
                    doc = json.loads(line)
                    doc_snippets = self.create_snippets(doc['text'])
                    
                    all_snippets.extend(doc_snippets)
                    all_doc_ids.extend([doc['doc_id']] * len(doc_snippets))
            
            print(f"Calculating embeddings for {len(all_snippets)} snippets...")
            embeddings = self.model.encode(all_snippets, show_progress_bar=True, batch_size=self.batch_size)
            embeddings = embeddings.astype(np.float16)
            
            print("Caching results...")
            self.snippets = all_snippets
            self.doc_ids = all_doc_ids
            cache_data = {
                'snippets': self.snippets,
                'doc_ids': self.doc_ids,
                'embeddings': embeddings
            }
            with open(snippets_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        
        if index_cache_file.exists():
            print("Loading cached index...")
            self.index = faiss.read_index(str(index_cache_file))
            print(f"Loaded index from cache")
        else:
            # Convert back to FP32 for FAISS (required)
            embeddings = embeddings.astype(np.float32)
            
            print("Building FAISS index...")
            dimension = embeddings.shape[1]
            num_snippets = len(self.snippets)

            # Check if GPU is available and use it for index building
            use_gpu = faiss.get_num_gpus() > 0
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            
            # Choose index type based on dataset size
            if num_snippets > 50000:
                # Calculate number of centroids based on power of 2 closest to sqrt of dataset size
                num_centroids = 8 * int(math.sqrt(math.pow(2, int(math.log(num_snippets, 2)))))
                print(f"Using {num_centroids} centroids for {num_snippets} snippets")
                
                self.index = faiss.index_factory(dimension, f"IVF{num_centroids}_HNSW32,Flat")
                
                if use_gpu:
                    print(f"Using {faiss.get_num_gpus()} GPUs for index building...")
                    index_ivf = faiss.extract_index_ivf(self.index)
                    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(dimension))
                    index_ivf.clustering_index = clustering_index
                
                print("Training index...")
                self.index.train(embeddings.astype(np.float32))
            else:
                print("Using simple FlatL2 index for small dataset...")
                self.index = faiss.IndexFlatL2(dimension)
                if use_gpu:
                    self.index = faiss.index_cpu_to_all_gpus(self.index)
            
            print("Adding vectors to index...")
            self.index.add(embeddings.astype(np.float32))
            
            # Set number of clusters to probe during search
            try:
                self.index.nprobe = 256
            except:
                pass
            
            print("Caching FAISS index...")
            faiss.write_index(self.index, str(index_cache_file))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve top-k relevant snippets with their document IDs and scores."""
        if self.index is None:
            raise ValueError("Index not built. Call process_corpus first.")
        
        #print(f"Processing query: {query}")
        # Encode and normalize query
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        #print(f"Searching for top {top_k} results...")
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Convert L2 distances to similarities
        similarities = [(2-d)/2 for d in distances[0]]
        
        # Return results
        results = []
        for similarity, idx in zip(similarities, indices[0]):
            results.append((
                self.doc_ids[idx],    # document ID
                self.snippets[idx],   # snippet text
                float(similarity)     # similarity score
            ))
        
        return results
parser = argparse.ArgumentParser()


parser.add_argument("--output_file", required=True)
parser.add_argument("--input_questions", required=True)
parser.add_argument("--retriever", type=str, default="Snowflake/snowflake-arctic-embed-l")
parser.add_argument("--input_plans", required=True)
parser.add_argument("--num_threads", type=int, default=1)
parser.add_argument("--n_retrieve", type=int, required=True)
parser.add_argument("--cache_dir", default="")
parser.add_argument("--corpus_path", required=True)


if __name__ == "__main__":
    
    opts = parser.parse_args()
    retriever = Retriever(cache_dir=opts.cache_dir, model_name=opts.retriever)
    retriever.process_corpus(opts.corpus_path)
    questions_dict = dict()
    with open(opts.input_questions) as file:
        try:
            questions = json.load(file)
        except:
            file.seek(0)
            questions = []
            for line in file:
                if line.strip():
                    obj = json.loads(line.strip())
                    questions.append(obj)
        for question in questions:
            questions_dict[question['query_id']] = question
    with open(opts.input_plans) as file:
        plans = json.load(file)
        for query_id, question in questions_dict.items():
            question['plans'] = plans[query_id]

    for query_id, query in questions_dict.items():
        for plan in query['plans']:
            for sub_q in plan['parsed_queries']:
                try:
                    q = sub_q['query']
                except:
                    print("Error in query")
                    sub_q['query'] = sub_q['aspect']
                    sub_q['reason'] = sub_q['aspect']
                    q = sub_q['query']
                ret_results = retriever.retrieve(q, opts.n_retrieve)
                ctx = []
                for doc in ret_results:
                    ctx.append(
                        {
                            "id": doc[0],
                            "title": "",
                            "text": doc[1],
                            "score": doc[2]
                        }
                    )
                sub_q['retrieved_docs'] = ctx
    
            
    os.makedirs(os.path.dirname(opts.output_file), exist_ok=True)

    with open(opts.output_file, "w") as file:
        json.dump([q for k, q in questions_dict.items()], file, indent = 4)