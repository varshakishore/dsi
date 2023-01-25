import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from transformers import T5Tokenizer, BertTokenizer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from dataclasses import dataclass
from T5Model import T5Model_projection
from BertModel import QueryClassifier
import random
import logging
logger = logging.getLogger(__name__)
import argparse
import os
import joblib
from utils import *
import faiss
import time


class DSIqgTrainDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            datadict):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['question'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        return input_ids, data['doc_id']
    



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings_path",
        # default="/home/jl3353/dsi/NQ320k_outputs/finetune_old_epoch17",
        default="/home/cw862/DSI/dsi/NQ320k_outputs/docs",
        type=str,
        help="path to saved model",
    )

    args = parser.parse_args()

    return args

def get_representations(embeddings_path):
    # Sentence embeddings and doc_ids
    old_doc_embeddings = joblib.load(os.path.join(embeddings_path, 'old_docs_embedding.pkl'))
    new_doc_embeddings = joblib.load(os.path.join(embeddings_path, 'new_docs_embedding.pkl'))
    train_embeddings = torch.cat((old_doc_embeddings, new_doc_embeddings), dim=0)
    
    old_doc_doc_ids = joblib.load(os.path.join(embeddings_path, 'old_docs-docids.pkl'))
    new_doc_doc_ids = joblib.load(os.path.join(embeddings_path, 'new_docs-docids.pkl'))
    train_doc_ids = torch.cat((old_doc_doc_ids, new_doc_doc_ids), dim=0)

    old_val_q_embeddings = joblib.load(os.path.join(embeddings_path, 'old-val-embeddings.pkl'))
    new_val_q_embeddings = joblib.load(os.path.join(embeddings_path, 'new-val-embeddings.pkl'))

    old_val_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'old-val-docids.pkl'))
    new_val_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'new-val-docids.pkl'))

    return train_embeddings, train_doc_ids, old_val_q_embeddings, old_val_q_doc_ids, new_val_q_embeddings, new_val_q_doc_ids

# def get_representations(embeddings_path):
#     # Sentence embeddings and doc_ids
#     old_gen_q_embeddings = joblib.load(os.path.join(embeddings_path, 'old-gen-embeddings.pkl'))
#     new_gen_q_embeddings = joblib.load(os.path.join(embeddings_path, 'new-gen-embeddings.pkl'))
#     old_q_embeddings = joblib.load(os.path.join(embeddings_path, 'old-train-embeddings.pkl'))
#     # new_q_embeddings = joblib.load(os.path.join(embeddings_path, 'tune-train-embeddings.pkl'))
#     train_embeddings = torch.cat((old_gen_q_embeddings, new_gen_q_embeddings, old_q_embeddings), dim=0)
    
#     old_gen_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'old-gen-docids.pkl'))
#     new_gen_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'new-gen-docids.pkl'))
#     old_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'old-train-docids.pkl'))
#     # new_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'tune-train-docids.pkl'))
#     train_doc_ids = torch.cat((old_gen_q_doc_ids, new_gen_q_doc_ids, old_q_doc_ids), dim=0)

#     old_val_q_embeddings = joblib.load(os.path.join(embeddings_path, 'old-val-embeddings.pkl'))
#     new_val_q_embeddings = joblib.load(os.path.join(embeddings_path, 'new-val-embeddings.pkl'))

#     old_val_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'old-val-docids.pkl'))
#     new_val_q_doc_ids = joblib.load(os.path.join(embeddings_path, 'new-val-docids.pkl'))

#     return train_embeddings, train_doc_ids, old_val_q_embeddings, old_val_q_doc_ids, new_val_q_embeddings, new_val_q_doc_ids

def validate(closest_docs, test_doc_ids, train_doc_ids):

    tok_opts = {}

    logger.info('Matching answers in top docs...')
    scores = []
    for query_idx in range(closest_docs.shape[0]): 
        doc_ids = [train_doc_ids[idx] for idx in closest_docs[query_idx]]
        hits = []
        for i, doc_id in enumerate(doc_ids):
            hits.append(test_doc_ids[query_idx] == doc_id)
        scores.append(hits)

    n_docs = len(closest_docs[0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    print('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(closest_docs) for v in top_k_hits]
    print('Validation results: top k documents hits accuracy %s', top_k_hits)
    return top_k_hits   

def  compute_mrr(closest_docs, test_doc_ids, train_doc_ids):

    logger.info('Matching answers in top docs...')
    mrr = 0.0
    for query_idx in range(closest_docs.shape[0]): 
        doc_ids = [train_doc_ids[idx] for idx in closest_docs[query_idx]]
        hits = []
        for i, doc_id in enumerate(doc_ids):
            # because we are computing mrr@10
            if i >= 10: break
            if test_doc_ids[query_idx] == doc_id:
                mrr += 1/(i+1)
                break
    mrr /= closest_docs.shape[0]
    print('Validation results mrr: %s', mrr)
    return mrr

def main():

    args = get_arguments()

    
    train_embeddings, train_doc_ids, old_test_embeddings, old_test_doc_ids, new_test_embeddings, new_test_doc_ids = get_representations(args.embeddings_path)

    dim = train_embeddings.shape[1]

    print("Adding to faiss index")
    start = time.time()
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(train_embeddings)
    print("Time taken to add to faiss index: ", time.time() - start)
    print("Done adding to faiss index")

    start = time.time()
    _, closest_docs = cpu_index.search(old_test_embeddings, 10)
    print("Time taken to search faiss index (old): ", time.time() - start)
    print("Done searching faiss index")
    top_k_hits = validate(closest_docs, old_test_doc_ids, train_doc_ids)
    mrr = compute_mrr(closest_docs, old_test_doc_ids, train_doc_ids)
    # import pdb; pdb.set_trace()

    start = time.time()
    _, closest_docs = cpu_index.search(new_test_embeddings, 10)
    print("Time taken to search faiss index (old): ", time.time() - start)
    top_k_hits = validate(closest_docs, new_test_doc_ids, train_doc_ids)
    mrr = compute_mrr(closest_docs, new_test_doc_ids, train_doc_ids)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()