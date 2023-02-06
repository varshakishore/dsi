import copy
import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from T5Model import T5Model_projection
from BertModel import QueryClassifier, DocClassifier
from dsi_model import DSIqgTrainDataset, GenPassageDataset 
import dsi_model_v1
from functools import partial
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, BertTokenizer
import pandas as pd
import datasets
import pickle as pkl
import joblib
import argparse
from utils import *


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default='T5-base',
        choices=['T5-base', 'bert-base-uncased'],
        help="Model name",
    )

    parser.add_argument(
        "--dataset", 
        default='nq320k_legacy', 
        choices=['nq320k_legacy', 'nq320k','msmarco'], 
        help='which dataset to use')

    parser.add_argument(
        "--initialize_model",
        default=None,
        type=str,
        help="path to saved model",
    )

    parser.add_argument(
        "--save_average",
        action="store_true",
        help="cache average embeddign ofr every class",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The dir. for log files",
    )

    parser.add_argument(
        "--doc_split",
        default = 'old',
        choices=['old','new', 'tune'],
        help="which split to save"
    )

    parser.add_argument(
        "--split",
        default = 'train',
        choices=['train','val', 'test', 'gen'],
        help="which split to save"
    )

    parser.add_argument(
        "--text_type",
        default = 'query',
        choices=['document','query'],
        help="save document or query embeddings"
    )
    
    args = parser.parse_args()

    return args

class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)
                
        inputs['labels'] = torch.Tensor(docids).long()
        return inputs



def save(args, model, dataloader, batch_size, dataset_size):

    model.eval()

    embedding = torch.zeros(dataset_size, 768)

    labels = torch.zeros(dataset_size)

    device = torch.device('cuda')

    for i,inputs in enumerate(tqdm(dataloader, desc='forward pass')):
                    
        inputs.to(device)            
        
        with torch.no_grad():
                            
            decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0],1))

            # print(inputs['labels'])

            if args.model_name == 'T5-base':
                _, outputs = model(input_ids=inputs['input_ids'].long(), decoder_input_ids=decoder_input_ids.long())
            elif args.model_name == 'bert-base-uncased':
                outputs = model(inputs['input_ids'], inputs['attention_mask'], return_hidden_emb=True)

            if i != len(dataloader) - 1:

                embedding[i*batch_size:(i+1)*batch_size] = outputs.squeeze()

                # import pdb; pdb.set_trace()
                labels[i*batch_size:(i+1)*batch_size] = inputs['labels']

            else:

                embedding[i*batch_size:i*batch_size+inputs['input_ids'].shape[0],:] = outputs.squeeze()
                labels[i*batch_size:i*batch_size+inputs['input_ids'].shape[0]] = inputs['labels']

    return embedding, labels



def main():

    print("hello")

    ######

    device = torch.device('cuda')

    args = get_arguments()

    ### HARDCODING 
    ### use the same number of class no matter which split to load because the output does not need the last layer
    class_num = 100000

    if args.text_type == 'query':
        if args.model_name == 'T5-base':
            model = T5Model_projection(class_num)
            tokenizer = T5Tokenizer.from_pretrained('t5-base',cache_dir='cache')
        elif args.model_name == 'bert-base-uncased':
            print('Using question model...')
            model = QueryClassifier(class_num)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')
    elif args.text_type == 'document':
        if args.model_name == 'T5-base':
            model = T5Model_projection(class_num)
            tokenizer = T5Tokenizer.from_pretrained('t5-base',cache_dir='cache')
        elif args.model_name == 'bert-base-uncased':
            print('Using ctx_model...')
            model = DocClassifier(class_num)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    if args.text_type == 'document' and args.dataset == 'nq320k':
        # HARDCODE 
        doc2class = joblib.load('/home/vk352/dsi/data/NQ320k/old_docs/doc_class.pkl')
        dataset_cls = partial(dsi_model_v1.DSIqgTrainDataset, doc_class=doc2class)
        file_path = '/home/cw862/DPR_data_final/NQ320k_docs.json'
        docs = datasets.load_dataset(
            'json',
            data_files=file_path,
            ignore_verifications=False,
            cache_dir='cache'
            )['train']

        queries = dataset_cls(tokenizer=tokenizer, datadict = docs)
        
    elif args.text_type == 'document' and args.dataset == 'msmarco':
        doc2class = joblib.load('/home/cw862/MSMARCO/old_docs/doc_class.pkl')
        dataset_cls = partial(dsi_model_v1.DSIqgTrainDataset, doc_class=doc2class)
        # TODO
        file_path = '/home/cw862/DPR_data_final/MSMARCO_new_docs.json'
        docs = datasets.load_dataset(
            'json',
            data_files=file_path,
            ignore_verifications=False,
            cache_dir='cache'
            )['train']

        queries = dataset_cls(tokenizer=tokenizer, datadict = docs)
        

    elif args.dataset == 'nq320k' and args.text_type == 'query':
        data_dirs = {'data': '/home/vk352/dsi/data/NQ320k',
                    'old': '/home/vk352/dsi/data/NQ320k/old_docs/',
                    'tune': '/home/vk352/dsi/data/NQ320k/tune_docs/',
                    'new': '/home/vk352/dsi/data/NQ320k/new_docs/'}
        if args.doc_split in ['old', 'new']:
            doc2class = joblib.load(os.path.join(data_dirs[args.doc_split], 'doc_class.pkl'))
        elif args.doc_split == 'tune':
            # Hardcoded path for tuning set
            doc2class = joblib.load('/home/jl3353/dsi/data/NQ320k/tune_docs/doc_class.pkl')
        else:
            raise ValueError(f'{args.doc_split} split not supported for {args.dataset} dataset')
        dataset_cls = partial(dsi_model_v1.DSIqgTrainDataset, doc_class=doc2class)
        gen_dataset_cls = partial(dsi_model_v1.GenPassageDataset, doc_class=doc2class)
    elif args.dataset == 'msmarco' and args.text_type == 'query':
        data_dirs = {'data': '/home/cw862/MSMARCO',
                    'old': '/home/cw862/MSMARCO/old_docs/',
                    'tune': '/home/cw862/MSMARCO/tune_docs/',
                    'new': '/home/cw862/MSMARCO/new_docs/'}
        if args.doc_split in ['old', 'new']:
            doc2class = joblib.load(os.path.join(data_dirs[args.doc_split], 'doc_class.pkl'))
        elif args.doc_split == 'tune':
            # TODO path for the doc_class for MSMARCO
            doc2class = joblib.load('/home/cw862/MSMARCO/tune_docs/doc_class.pkl')
        else:
            raise ValueError(f'{args.doc_split} split not supported for {args.dataset} dataset')
        dataset_cls = partial(dsi_model_v1.DSIqgTrainDataset, doc_class=doc2class)
        gen_dataset_cls = partial(dsi_model_v1.GenPassageDataset, doc_class=doc2class)
        
    else:
        raise ValueError(f'{args.dataset} dataset not supported')

    if args.text_type == 'query':
        if args.split == 'gen':
            file_path = os.path.join(data_dirs[args.doc_split], 'passages_seen.json')
            
            generated_queries = datasets.load_dataset(
            'json',
            data_files=file_path,
            ignore_verifications=False,
            cache_dir='cache'
            )['train'] 

            queries = gen_dataset_cls(tokenizer=tokenizer, datadict = generated_queries)
        else:
            
            file_path = os.path.join(data_dirs[args.doc_split], f'{args.split}queries.json')
            natural_queries = datasets.load_dataset(
            'json',
            data_files=file_path,
            ignore_verifications=False,
            cache_dir='cache'
            )['train']

            queries = dataset_cls(tokenizer=tokenizer, datadict = natural_queries)

    dataloader = DataLoader(queries, 
                            batch_size=3500,
                            collate_fn=IndexingCollator(
                            tokenizer,
                            padding='longest'),
                            shuffle=False,
                            drop_last=False)


    # load saved model if initialize_model
    # TODO extend for T5
    if args.initialize_model:
        load_saved_weights(model, args.initialize_model, strict_set=False, load_classifier=False)

    model = torch.nn.DataParallel(model)
    model.to(device)
    embedding_matrix, labels = save(args, model, dataloader, 3500, len(queries))
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.save_average:
        embedding_matrix_pd = pd.DataFrame(embedding_matrix)
        embedding_matrix_pd.insert(0, "doc_ids", labels)
        embedding_matrix_avg = embedding_matrix_pd.groupby('doc_ids').mean()

        #### TODO remove both the hardcodings below
        assert (embedding_matrix_avg.index.values == [i for i in range(109715)]).all()

        with open('/home/cw862/DSI/data/nq320k_passagesembedding.pkl', 'wb') as f:
            pkl.dump(embedding_matrix_avg, f)

        print('embedding file written.')

    elif args.text_type == 'query':
        print(f'Writing {args.doc_split}-{args.split}-embeddings.pkl')
        joblib.dump(embedding_matrix, os.path.join(args.output_dir,f'{args.doc_split}-{args.split}-embeddings.pkl'))
        print('Done.')
        class2doc = {v:k for k, v in doc2class.items()}
        assert len(class2doc) == len(doc2class)
        doc_ids = torch.tensor([class2doc[i.item()] for i in labels], dtype=torch.long)
        print(f'Writing {args.doc_split}-{args.split}-docids.pkl')
        joblib.dump(doc_ids, os.path.join(args.output_dir, f'{args.doc_split}-{args.split}-docids.pkl'))
        print('Done.')

    elif args.text_type == 'document':
        joblib.dump(embedding_matrix, os.path.join(args.output_dir, 'new_document_embedding.pkl'))
        class2doc = {v:k for k, v in doc2class.items()}
        assert len(class2doc) == len(doc2class)
        doc_ids = torch.tensor([class2doc[i.item()] for i in labels], dtype=torch.long)
        joblib.dump(doc_ids, os.path.join(args.output_dir, f'new_document-docids.pkl'))


if __name__ == "__main__":
    main()
  




            
            


