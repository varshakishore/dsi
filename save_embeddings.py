import copy
import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from T5Model import T5Model_projection
from BertModel import QueryClassifier
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

    args = parser.parse_args()

    return args



class passagesloader(Dataset):
    def __init__(
            self,
            tokenizer,
            datadict):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['gen_question'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        return input_ids, data['doc_id']

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

    for i,inputs in tqdm(enumerate(dataloader), desc='forward pass'):
                    
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

    print("hallo")

    ######

    device = torch.device('cuda')

    args = get_arguments()

    generated_queries = datasets.load_dataset(
    'json',
    data_files='/home/vk352/ANCE/NQ320k_dataset_v2/passages.json',
    ignore_verifications=False,
    cache_dir='cache'
    )['train'] 

    class_num=int(len(generated_queries)/10)

    if args.model_name == 'T5-base':
        model = T5Model_projection(class_num)
        tokenizer = T5Tokenizer.from_pretrained('t5-base',cache_dir='cache')
    elif args.model_name == 'bert-base-uncased':
        model = QueryClassifier(class_num)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    # load saved model if initialize_model
    # TODO extend for T5
    if args.initialize_model:
        load_saved_weights(model, args.initialize_model, strict_set=False, load_classifier=False)

    model = torch.nn.DataParallel(model)
    model.to(device)


    gen_queries = passagesloader(tokenizer=tokenizer, datadict = generated_queries )

    dataloader = DataLoader(gen_queries, 
                                batch_size=3500,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=False,
                                drop_last=False)

    embedding_matrix, labels = save(args, model, dataloader, 3500, len(generated_queries))

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
        # check if order of samples is 0,0,0,0,0,0,0,0,0,0,1,1,...,109715,109715
        (labels.numpy() == [x for x in range(109715) for y in range(10)]).all()

        import pdb; pdb.set_trace()
        joblib.dump(embedding_matrix, args.output_dir)
        # with open(args.output_dir, 'wb') as f:
        #     pkl.dump(embedding_matrix, f)

        print('embedding file written.')



if __name__ == "__main__":
    main()
  




            
            

