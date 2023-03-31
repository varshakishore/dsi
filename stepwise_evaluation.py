import datasets
from transformers import BertTokenizer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from BertModel import QueryClassifier
import random
import logging
logger = logging.getLogger(__name__)
import argparse
import os
import joblib
import json
from utils import *
from direct_optimize import initialize_model, initialize_nq320k
from dsi_model_v1 import validate_script


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=2000,
        type=int,
        required=False,
        help="batch_size",
    )

    parser.add_argument(
        "--model_name",
        default='bert-base-uncased',
        choices=['T5-base', 'bert-base-uncased'],
        help="Model name",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
        help="random seed",
    )

    parser.add_argument(
        "--initialize_embeddings",
        default=None,
        type=str,
        help="folder for the embedding matrix",
    )

    parser.add_argument(
        "--initialize_model",
        default=None,
        type=str,
        help="path to saved model",
    )

    parser.add_argument(
        "--eval_step",
        default=1000,
        type=int,
        help="step for evaluation"
    )

    parser.add_argument(
        "--dataset", 
        default='NQ320k', 
        choices=['NQ320k', 'MSMARCO'], 
        help='which dataset to use')

    parser.add_argument(
        "--write_path",
        default=None,
        type=str,
        help='folder to write the results'
    )

    parser.add_argument(
        "--doc_type",
        default=None,
        type=str,
        choices=['old_docs','new_docs','tune_docs'],
        help='which doc split to use'
    )

    parser.add_argument(
        "--base_data_dir",
        default=None,
        type=str,
        help='path to dataset'
    )

    parser.add_argument(
        "--frequent",
        action='store_true',
        help='run frequent evaluation'
    )

    args = parser.parse_args()

    return args


def Getmodel(args, embedding_matrix, class_num):    

    print(f'Loading Model and Tokenizer for {args.model_name}')

    model = QueryClassifier(class_num)
    
    load_saved_weights(model, args.initialize_model, strict_set=False)

    ### Use pre_calculated weights to initialize the projection matrix
    model.classifier.weight.data = embedding_matrix[:class_num].detach().to('cpu')  

    device = torch.device("cuda") 

    model = torch.nn.DataParallel(model)
    model.to(device)

    print(f'Device: {device}')

    print('model loaded')

    return model


def main():

    args = get_arguments()

    if args.model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    if args.dataset == "NQ320k":
        old_class_num = 98743
    elif args.dataset == "MSMARCO":
        old_class_num = 289424
    else:
        raise ValueError(f'dataset={args.dataset} must be NQ320k or MSMARCO')
    
    embedding_matrix = joblib.load(os.path.join(args.initialize_embeddings, 'classifier_layer.pkl'))

    if args.frequent:
        with open(os.path.join(args.initialize_embeddings, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
            if 'permutation_seed' in saved_args:
                permutation_seed = saved_args['permutation_seed']
            else:
                permutation_seed = None
            print(permutation_seed)
        eval_doc_nums = [10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        results = {'Num_Documents':[], 'Doc_type':[], 'Split':[], 'H@1':[], 'H@5':[], 'H@10':[], 'MRR@10':[]}
        for num_eval_doc in eval_doc_nums:
            class_num = old_class_num + num_eval_doc

            model = Getmodel(args, embedding_matrix, class_num)

            doc_types = ['old','new']
            if num_eval_doc < 1000:
                split = 'valid'
            else:
                split = 'test'
            
            for doc_type in doc_types:
                if doc_type == 'new':
                    if permutation_seed is not None:
                        filter_docs_list = joblib.load(os.path.join('/home/jl3353/dsi/data/NQ320k/new_docs', f'doc_list_seed{permutation_seed}.pkl'))
                    else:
                        filter_docs_list = joblib.load(os.path.join(args.base_data_dir, f'{doc_type}_docs', 'doc_list.pkl'))
                    filter_docs_list = filter_docs_list[:num_eval_doc]
                    h1, h5, h10, mrr_10 = validate_script(args, tokenizer, model, doc_type, split, filter_docs_list, permutation_seed)
                else:
                    h1, h5, h10, mrr_10 = validate_script(args, tokenizer, model, doc_type, split)
                results['Num_Documents'].append(num_eval_doc)
                results['Split'].append(split)
                results['Doc_type'].append(doc_type)
                results['H@1'].append(h1.item())
                results['H@5'].append(h5.item())
                results['H@10'].append(h10.item())
                results['MRR@10'].append(mrr_10.item())
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(args.write_path, 'results.csv'), index=False)
        return

                
            
    
    eval_doc_nums = [10, 100, 1000, 10000]


    print('query embeddings loaded')


    results_list = []

    for num_eval_doc in eval_doc_nums:
        class_num = old_class_num + num_eval_doc

        model = Getmodel(args, embedding_matrix, class_num)

        doc_types = ['old','new']
        splits = ['valid', 'test']
        results = {}
        for doc_type in doc_types:
            if doc_type == 'new':
                filter_docs_list = joblib.load(os.path.join(args.base_data_dir, f'{doc_type}_docs', 'doc_list.pkl'))
                filter_docs_list = filter_docs_list[:num_eval_doc]
            for split in splits:
                print(doc_type, split)
                if doc_type == 'new':
                    h1, h5, h10, mrr_10 = validate_script(args, tokenizer, model, doc_type, split, filter_docs_list)
                else:
                    h1, h5, h10, mrr_10 = validate_script(args, tokenizer, model, doc_type, split)
                results[f'{doc_type}_{split}'] = [h1, h5, h10, mrr_10]
                print(f'At step {num_eval_doc}:')
                print(f'{doc_type}_{split}: {results[f"{doc_type}_{split}"]}')
        results_list.append(results)
        
        
    if not os.path.exists(args.write_path):
        os.mkdir(args.write_path)

    timeslist = joblib.load(os.path.join(args.initialize_embeddings, 'timelist.pkl'))

    with open(os.path.join(args.write_path, 'eval_results.txt'),'w') as results_file:
        for num_eval_doc, results in zip(eval_doc_nums, results_list):
            results_file.write(f'num_eval_doc: {num_eval_doc}' + '\n')
            results_file.write(f'time to add: {np.asarray(timeslist[:num_eval_doc]).sum()}' + '\n')
            for doc_type in doc_types:
                for split in splits:
                    assert len(results[f"{doc_type}_{split}"]) == 4
                    results_file.write(f'{doc_type}_{split}' + '\n')
                    results_file.write(f'hits@1: {results[f"{doc_type}_{split}"][0]}' + '\n')
                    results_file.write(f'hits@5: {results[f"{doc_type}_{split}"][1]}' + '\n')
                    results_file.write(f'hits@10: {results[f"{doc_type}_{split}"][2]}' + '\n')
                    results_file.write(f'mrr@10: {results[f"{doc_type}_{split}"][3]}' + '\n\n')

    print(f'results written.')

if __name__ == "__main__":
    main()

    






