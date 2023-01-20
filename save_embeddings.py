import copy
import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from T5Model import T5Model_projection
from BertModel import QueryClassifier
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
        "--split",
        default = 'gen',
        choices=['train','val', 'test', 'gen'],
        help="which split to save"
    )
    
    parser.add_argument('--generated', default=False, action='store_true')

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

    if args.model_name == 'T5-base':
        model = T5Model_projection(class_num)
        tokenizer = T5Tokenizer.from_pretrained('t5-base',cache_dir='cache')
    elif args.model_name == 'bert-base-uncased':
        model = QueryClassifier(class_num)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    if args.dataset == 'nq320k_legacy':
        data_dirs = {'gen': '/home/vk352/ANCE/NQ320k_dataset_v2/passages.json',
                    'train': '/home/vk352/ANCE/NQ320k_dataset_v2/trainqueries_extended.json',
                    'val': '/home/vk352/ANCE/NQ320k_dataset_v2/testqueries.json'}
        dataset_cls = DSIqgTrainDataset
        gen_dataset_cls = GenPassageDataset
    elif args.dataset == 'nq320k':
        data_dirs = {'data': '/home/vk352/dsi/data/NQ320k',
                    'train': '/home/vk352/dsi/data/NQ320k/old_docs/trainqueries.json',
                    'val': '/home/vk352/dsi/data/NQ320k/tune_docs/trainqueries.json',
                    'test': '/home/vk352/dsi/data/NQ320k/new_docs/trainqueries.json'}
        if args.split in ['train', 'test']:
            doc2class = joblib.load(os.path.join(os.path.dirname(data_dirs[args.split]), 'doc_class.pkl'))
        elif args.split == 'val':
            # Hardcoded path for tuning set
            doc2class = joblib.load('/home/jl3353/dsi/data/NQ320k/tune_docs/doc_class.pkl')
        else:
            raise ValueError(f'{args.split} split not supported for {args.dataset} dataset')
        dataset_cls = partial(dsi_model_v1.DSIqgTrainDataset, doc_class=doc2class)
        gen_dataset_cls = partial(dsi_model_v1.GenPassageDataset, doc_class=doc2class)
    elif args.dataset == 'msmarco':
        data_dirs = {'data': '/home/cw862/MSMARCO',
                    'train': '/home/cw862/MSMARCO/old_docs/trainqueries.json',
                    'val': '/home/cw862/MSMARCO/tune_docs/trainqueries.json',
                    'test': '/home/cw862/MSMARCO/new_docs/trainqueries'}
        if args.split in ['train', 'test']:
            doc2class = joblib.load(os.path.join(os.path.dirname(data_dirs[args.split]), 'doc_class.pkl'))
        elif args.split == 'val':
            # TODO path for the doc_class for MSMARCO
            doc2class = joblib.load('/home/cw862/MSMARCO/tune_docs/doc_class.pkl')
        else:
            raise ValueError(f'{args.split} split not supported for {args.dataset} dataset')
        dataset_cls = partial(dsi_model_v1.DSIqgTrainDataset, doc_class=doc2class)
        gen_dataset_cls = partial(dsi_model_v1.GenPassageDataset, doc_class=doc2class)
        
    else:
        raise ValueError(f'{args.dataset} dataset not supported')


    if args.generated:
        file_path = os.path.join(os.path.dirname(data_dirs[args.split]), 'passages_seen.json')
        
        generated_queries = datasets.load_dataset(
        'json',
        data_files=file_path,
        ignore_verifications=False,
        cache_dir='cache'
        )['train'] 

        queries = gen_dataset_cls(tokenizer=tokenizer, datadict = generated_queries)
    elif args.split == 'gen':
        assert args.dataset == 'nq320k_legacy', 'only legacy dataset is supported'
        queries = datasets.load_dataset(
        'json',
        data_files='/home/vk352/ANCE/NQ320k_dataset_v2/passages.json',
        ignore_verifications=False,
        cache_dir='cache'
        )['train']

        queries = gen_dataset_cls(tokenizer=tokenizer, datadict = queries)

    elif args.split in ['train', 'val', 'test']:
        queries = datasets.load_dataset(
        'json',
        data_files=data_dirs[args.split],
        ignore_verifications=False,
        cache_dir='cache'
        )['train']

        queries = dataset_cls(tokenizer=tokenizer, datadict = queries)
    else:
        raise ValueError(f'{args.split} split not supported')        

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

    else:
        if args.generated:

        # check if order of samples is 0,0,0,0,0,0,0,0,0,0,1,1,...,109715,109715
            if args.dataset == 'nq320k_legacy':
                assert (labels.numpy() == [x for x in range(109715) for y in range(10)]).all()
                joblib.dump(embedding_matrix, args.output_dir)
            elif args.dataset == 'nq320k' or args.dataset == "msmarco":
                joblib.dump(embedding_matrix, os.path.join(args.output_dir,f'{args.split}-gen-embeddings.pkl'))
                class2doc = {v:k for k, v in doc2class.items()}
                assert len(class2doc) == len(doc2class)
                doc_ids = torch.tensor([class2doc[i.item()] for i in labels], dtype=torch.long)
                joblib.dump(doc_ids, os.path.join(args.output_dir, f'{args.split}-gen-docids.pkl'))
            else:
                raise ValueError(f'{args.dataset} dataset not supported')

            
        # with open(args.output_dir, 'wb') as f:
        #     pkl.dump(embedding_matrix, f)

            print('embedding file written.')

        else:
            if args.dataset == 'nq320k_legacy':
                embedding_matrix_pd = pd.DataFrame(embedding_matrix)
                embedding_matrix_pd.insert(0, "doc_ids", labels)
                embedding_matrix_sorted = embedding_matrix_pd.sort_values(by=['doc_ids'])
                import pdb;pdb.set_trace()
                ### the index of sorting 
                index = embedding_matrix_sorted.index
                ### doc_ids list, access through the index
                doc_ids = embedding_matrix_sorted['doc_ids']

                joblib.dump(index, os.path.join(args.output_dir,f'{args.split}-index.pkl'))
                joblib.dump(doc_ids, os.path.join(args.output_dir, f'{args.split}-docids.pkl'))
                joblib.dump(embedding_matrix, os.path.join(args.output_dir,f'{args.split}-embeddings.pkl'))

                print('embedding matrix, index and docids written.')
            elif args.dataset == 'nq320k' or args.dataset == 'msmarco':
                joblib.dump(embedding_matrix, os.path.join(args.output_dir,f'{args.split}-embeddings.pkl'))
                class2doc = {v:k for k, v in doc2class.items()}
                assert len(class2doc) == len(doc2class)
                doc_ids = torch.tensor([class2doc[i.item()] for i in labels], dtype=torch.long)
                joblib.dump(doc_ids, os.path.join(args.output_dir, f'{args.split}-docids.pkl'))
            else:
                raise ValueError(f'{args.dataset} dataset not supported')


if __name__ == "__main__":
    main()
  




            
            


