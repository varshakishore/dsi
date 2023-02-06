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
from utils import *
from dsi_model_continual import DSIqgTrainDataset, GenPassageDataset, IndexingCollator
from direct_optimize import initialize_model, initialize_nq320k
from dsi_model_v1 import validate


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
        "--data_path",
        default=None,
        type=str,
        help='path to dataset'
    )

    args = parser.parse_args()

    return args

def loaddataset(args, tokenizer):

    seen_gen_qs = datasets.load_dataset(
        'json',
        data_files=os.path.join(args.data_path, args.doc_type, 'passages_seen.json'),
        ignore_verifications=False,
        cache_dir='cache'
    )['train']   

    unseen_gen_qs = datasets.load_dataset(
        'json',
        data_files=os.path.join(args.data_path, args.doc_type, 'passages_unseen.json'),
        ignore_verifications=False,
        cache_dir='cache'
    )['train']

    print(f'passages loaded with length {len(seen_gen_qs)}')
    print(f'unseen passages loaded with length {len(unseen_gen_qs)}')

    val_data = datasets.load_dataset(
    'json',
    data_files=os.path.join(args.data_path, args.doc_type, 'valqueries.json'),
    ignore_verifications=False,
    cache_dir='cache'
    )['train']

    print(f'validation set loaded with length {len(val_data)}')

    test_data = datasets.load_dataset(
    'json',
    data_files=os.path.join(args.data_path, args.doc_type, 'testqueries.json'),
    ignore_verifications=False,
    cache_dir='cache'
    )['train']

    print(f'test set loaded with length {len(test_data)}')
    
    print('datasets loaded')

    if args.doc_type == "new_docs" or args.doc_type == "old_docs":
        doc_class = joblib.load(os.path.join(args.data_path, 'new_docs', 'doc_class.pkl'))
    elif args.doc_type == "tune_docs":
        doc_class = joblib.load(os.path.join(args.data_path, 'tune_docs', 'doc_class.pkl'))

    val_dataset =  DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data, doc_class = doc_class)
    test_dataset = DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data, doc_class = doc_class)

    seenq_dataset = GenPassageDataset(tokenizer=tokenizer, datadict = seen_gen_qs, doc_class=doc_class)
    unseenq_dataset = GenPassageDataset(tokenizer=tokenizer, datadict = unseen_gen_qs, doc_class=doc_class)

    return [val_dataset, test_dataset, seenq_dataset, unseenq_dataset]
    
def filter_Dataloader(args, old_class_num, class_num, datasets, tokenizer):

    new_class_num = class_num - old_class_num
    print(f'Old class number: {old_class_num}')
    print(f'New class number: {new_class_num}')
    print(f'Total class number:  {class_num}')

    # use the data only till the class_num docs
    print('Filtering datasets')

    filtered_datasets = [dataset.filter(lambda example: example[1] < class_num) for dataset in datasets]

    new_datasets = [dataset.filter(lambda example: example[1] >= old_class_num) for dataset in filtered_datasets]

    old_datasets = [dataset.filter(lambda example: example[1] < old_class_num) for dataset in filtered_datasets]

    print(f"Filtered set:")
    print('\n')
    print(f"Old-Val-{len(old_datasets[0])}, New-Val-{len(new_datasets[0])}") 
    print(f"Old-Test-{len(old_datasets[1])}, New-Test-{len(new_datasets[1])}") 
    print(f"seen queries from old docs{len(old_datasets[2])}, seen queries from new docs{len(new_datasets[2])}") 
    print(f"unseen queries from old docs{len(old_datasets[3])}, unseen queries from new docs{len(new_datasets[3])}") 

    old_dataloaders = [DataLoader(old_data, batch_size=args.batch_size,collate_fn=IndexingCollator(tokenizer,padding='longest'),shuffle=False,drop_last=False)
                    for old_data in old_datasets]

    new_dataloaders = [DataLoader(new_data, batch_size=args.batch_size,collate_fn=IndexingCollator(tokenizer,padding='longest'),shuffle=False,drop_last=False)
                    for new_data in new_datasets]

    return old_dataloaders, new_dataloaders

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

    datasets = loaddataset(args.data_path, args.doc_type)

    embedding_matrix = joblib.load(os.path.join(args.initialize_embeddings, 'classifier_layer.pkl'))

    print('query embeddings loaded')


    h1 = []
    h5 = []
    h10 = []
    m10 = []
    progress = []

    for class_num in range(old_class_num, embedding_matrix.shape[0], args.eval_step):

        added_num = class_num - old_class_num

        progress.append(added_num)

        model = Getmodel(args, embedding_matrix, class_num)

        old_dataloaders, new_dataloaders = filter_Dataloader(args, old_class_num, class_num, datasets, tokenizer)

        dataloaders = old_dataloaders.extend(new_dataloaders)

        hits1 = []
        hits5 = []
        hits10 = []
        mrr10 = []

        splits = ['val queries for old docs', 'test queries for old docs', 'seen generated queries for old docs','unseen generated queries for old docs',
                  'val queries for new docs', 'test queries for new docs', 'seen generated queries for new docs','unseen generated queries for new docs']
        
        for i,split in enumerate(dataloaders):
            print('*'*100)
            print(splits[i])
            acc1, acc5, acc10, mrr_10 = validate(args, model, split)
            hits1.append(acc1.item())
            hits5.append(acc5.item())
            hits10.append(acc10.item())
            mrr10.append(mrr_10.item())

        print(f'At step {added_num}:')
        print(splits)

        print(f'hits@1: {hits1}')
        print(f'hits@5: {hits5}')
        print(f'hits@10: {hits1}')
        print(f'mrr@10: {mrr10}')

        h1.append(hits1)
        h5.append(hits5)
        h10.append(hits10)
        m10.append(mrr10)

    if not os.path.exists(args.write_path):
        os.mkdir(args.write_path)

    with open(os.path.join(args.write_path, 'eval_results.txt'),'w') as results:
        results.write(f'Evaluation steps: {progress}' + '\n')
        results.write(f'splits: {splits}')
        results.write(f'hits@1: {h1}' + '\n')
        results.write(f'hits@5: {h5}' + '\n')
        results.write(f'hits@10: {h10}' + '\n')
        results.write(f'mrr@10: {m10}' + '\n')

    print(f'results written.')

if __name__ == "__main__":
    main()

    






