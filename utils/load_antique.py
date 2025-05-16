import ir_datasets
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--train_addr", type=str, required=True)
parser.add_argument("--test_addr", type=str, required=True)
parser.add_argument("--corpus_addr", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    train_addr = args.train_addr
    test_addr = args.test_addr
    corpus_addr = args.corpus_addr
    
    dataset_train = ir_datasets.load('antique/train')
    dataset_test = ir_datasets.load('antique/test')

    train_list = []
    test_list = []

    for query in dataset_train.queries_iter():
        train_list.append(
            {
                "query_id": query.query_id,
                "query_description": query.text,
            }
        )

    for query in dataset_test.queries_iter():
        test_list.append(
            {
                "query_id": query.query_id,
                "query_description": query.text,
            }
        )

    corpus = []

    for doc in dataset_train.docs_store():
        if len(doc.text.split()) < 50:
            continue
        corpus.append(
            {
                "doc_id": doc.doc_id,
                "title": "",
                "text": doc.text,
            }
        )
    with open(train_addr, "w") as f:
        for item in train_list:
            f.write(json.dumps(item) + "\n")

    with open(test_addr, "w") as f:
        for item in test_list:
            f.write(json.dumps(item) + "\n")

    with open(corpus_addr, "w") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")
