import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from dataclasses import dataclass
import random
import logging
logger = logging.getLogger(__name__)
import argparse
import os
import joblib
from utils import *
from transformers import AdamW

import wandb

id2token_map = {}
global_step = 0

class DSIqgTrainDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            datadict,
            doc_class):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['question'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        doc_id_str = str(self.doc_class[data['doc_id']])

        doc_ids = self.tokenizer.encode(list(doc_id_str),
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=8, padding='max_length') 
        
        return input_ids, doc_ids, doc_id_str

class DSIqgDocDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            datadict,
            doc_class):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['doc_text'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        doc_id_str = str(self.doc_class[data['doc_id']])

        doc_ids = self.tokenizer.encode(list(doc_id_str),
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=8, padding='max_length') 
        
        return input_ids, doc_ids, doc_id_str
    

class GenPassageDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            datadict,
            doc_class):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['gen_question'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        doc_id_str = str(self.doc_class[data['doc_id']])

        doc_ids = self.tokenizer.encode(list(doc_id_str),
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=8, padding='max_length') 
        
        return input_ids, doc_ids, doc_id_str

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=50,
        type=int,
        required=False,
        help="batch_size",
    )

    parser.add_argument(
        "--train_epochs",
        default=128,
        type=int,
        help="Number of train epochs",
    )

    parser.add_argument(
        "--model_name",
        default='t5-base',
        choices=['t5-base'],
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
        "--learning_rate",
        default=5e-3,
        type=float,
        help="initial learning rate for Adam",
    )

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="only runs validaion",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The dir. for log files",
    )

    parser.add_argument(
        "--logging_step",
        default=50,
        type=int,
        required=False,
        help="steps to log train loss and accuracy"
    )

    parser.add_argument(
        "--initialize_embeddings",
        default=None,
        type=str,
        help="file for the embedding matrix",
    )

    parser.add_argument(
        "--ance_embeddings",
        action="store_true",
        help="are these embeddings from ance",
    )

    parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        help="for freezing the parameters of the base model",
    )

    parser.add_argument(
        "--initialize_model",
        default=None,
        type=str,
        help="path to saved model",
    )

    parser.add_argument(
        "--base_data_dir_new",
        default=None,
        help="finetune with old and new documents",
    )

    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="/home/vk352/dsi/data/NQ320k",
        help="where the train/test/val data is located",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="finetune_old_epoch",
        help="name for savecd model",
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
        help="run eval on test set",
    )

    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="name for wandb",
    )
    
    parser.add_argument(
        "--doc_index",
        action="store_true",
        help="like dsi perform indexing with doc text as well",
    )

    parser.add_argument(
        "--get_subsampled_train_dataloader",
        action="store_true",
        help="randomly sample equal number of old docs",
    )

    parser.add_argument(
        "--filter_num",
        type=int,
        default=-1,
        help="num new docs",
    )

    parser.add_argument(
        "--new_only",
        action="store_true",
        help="train with new queries only",
    )
    

    args = parser.parse_args()

    return args



@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        docid_str = [x[2] for x in features]
        inputs = super().__call__(input_ids)
                
        inputs['labels'] = torch.cat(docids)
        # set labels to -100 for padding tokens
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        inputs['docid_str'] = docid_str
        
        return inputs

def train(args, model, train_dataloader, optimizer, length, scheduler=None, i=0):
    global global_step

    model.train()

    device = torch.device('cuda')
    tr_loss = torch.tensor(0.0).to(device)

    # if i==1: import pdb; pdb.set_trace()

    for i, inputs in enumerate(train_dataloader):

        del inputs['docid_str']
        inputs.to(device)

        loss = model(input_ids=inputs['input_ids'].long(), attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        # if i%50 == 0:
        #     import pdb; pdb.set_trace()
        
        
        if (i) % args.logging_step == 0 and i > 0:
            logger.info(f'Train step: {i}, loss: {(tr_loss/args.logging_step).item()}, Global step: {global_step}')
            if args.wandb_name:
                wandb.log({'train_loss': (tr_loss/args.logging_step).item()})
                wandb.log({'learning_rate': scheduler.get_last_lr()[0]})
                wandb.log({'learning_rate_cp': scheduler.get_last_lr()[0]}, step=global_step)
            
            tr_loss = torch.tensor(0.0).to(device)

        tr_loss += loss

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1

        # global_step += 1
    
    logger.info(f'Loss:{tr_loss/(i+1)}')
    return tr_loss

def validate(args, model, val_dataloader, restrict_decode_vocab, epoch):
    if epoch == 0:
        return 0, 0, 0, 0

    model.eval()

    hit_at_1 = 0
    hit_at_10 = 0
    mrr_at_10 = 0
    hit_at_5 = 0
    val_loss = 0

    device = torch.device('cuda')

    if epoch == 1:
        import pdb; pdb.set_trace()


    for i,inputs in tqdm(enumerate(val_dataloader), desc='Evaluating dev queries'):
                    
        labels_str = np.asarray(inputs['docid_str'])
        del inputs['docid_str']
        inputs.to(device)
        
        with torch.no_grad():
                            
            batch_beams = model.module.generate(
                    inputs['input_ids'],
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True, )

        
            rank_list = np.vectorize(id2token_map.__getitem__)(batch_beams.cpu())
            rank_list = np.apply_along_axis(''.join, 1, rank_list).reshape(inputs['input_ids'].shape[0], 10)

            hit_at_1 += (labels_str == rank_list[:, 0]).sum()
            print(i, hit_at_1)
            
            labels_str = np.expand_dims(labels_str, axis=1)

            hit_at_5 += (labels_str == rank_list[:, :5]).any(1).sum()

            hit_at_10 += (labels_str == rank_list).any(1).sum()

            #compute mrr@10. Sum will later be divided by number of elements
            mrr_at_10 += (1/ (np.where(labels_str == rank_list)[1] + 1)).sum()

    if epoch == 1:
        import pdb; pdb.set_trace()
            
    return hit_at_1, hit_at_5, hit_at_10, mrr_at_10

def validate_script(args, tokenizer, model, doc_type=None, split=None):

    device = torch.device("cuda")

    logging.info(f'Device: {device}')

    if doc_type == "old":
        data_dir = os.path.join(args.base_data_dir, 'old_docs')
        doc_class = joblib.load(os.path.join(data_dir, 'doc_class.pkl'))
    elif doc_type == "new":
        data_dir = os.path.join(args.base_data_dir, 'new_docs')
        doc_class = joblib.load(os.path.join(data_dir, 'doc_class.pkl'))
        if 'MSMARCO' in args.base_data_dir:
            doc_list = joblib.load(os.path.join(data_dir, 'doc_list.pkl'))
            doc_list = doc_list[:10000]
    elif doc_type == "tune":
        data_dir = os.path.join(args.base_data_dir, 'tune_docs')
        doc_class = joblib.load(os.path.join(data_dir, 'doc_class.pkl'))
        if 'MSMARCO' in args.base_data_dir:
            doc_list = joblib.load(os.path.join(data_dir, 'doc_list.pkl'))
            doc_list = doc_list[:1000]
    else:
        raise ValueError(f'doc_type={doc_type} must be old, new, or tune')

    if split == "train":

        data = datasets.load_dataset(
        'json',
        data_files=os.path.join(data_dir, 'trainqueries.json'),
        ignore_verifications=False,
        cache_dir='cache'
        )['train']

        print('train set loaded')

    elif split == "valid":
        data = datasets.load_dataset(
        'json',
        data_files=os.path.join(data_dir, 'valqueries.json'),
        ignore_verifications=False,
        cache_dir='cache'
        )['train']

        print('validation set loaded')

    elif split == "test":
        data = datasets.load_dataset(
        'json',
        data_files=os.path.join(data_dir, 'testqueries.json'),
        ignore_verifications=False,
        cache_dir='cache'
        )['train']

        print('test set loaded')

    elif split == "seenq":
        data = datasets.load_dataset(
            'json',
            data_files=os.path.join(data_dir, 'passages_seen.json'),
            ignore_verifications=False,
            cache_dir='cache'
        )['train']   

        print('seen generated queries loaded')

    elif split == "unseenq":
        data = datasets.load_dataset(
        'json',
        data_files=os.path.join(data_dir, 'passages_unseen.json'),
        ignore_verifications=False,
        cache_dir='cache'
        )['train']

        print('unseen generated queries loaded')
    else:
        raise ValueError(f'split={split} must be train, valid, test, seenq, or unseenq')
    
    if 'MSMARCO' in args.base_data_dir and doc_type in {'new', 'tune'}:
        data = data.filter(lambda example: example['doc_id'] in doc_list)

    if split == "train" or split == "valid" or split == 'test':
        dataset =  DSIqgTrainDataset(tokenizer=tokenizer, datadict = data, doc_class = doc_class)

    elif split == "seenq" or split == "unseenq":
        dataset = GenPassageDataset(tokenizer=tokenizer, datadict = data, doc_class = doc_class)

    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size,
                            collate_fn=IndexingCollator(
                            tokenizer,
                            padding='longest'),
                            shuffle=False,
                            drop_last=False)

    hits_at_1, hits_at_5, hits_at_10, mrr_at_10 = validate(args, model, dataloader)
    length = len(dataloader.dataset)
    hits_at_1 = hits_at_1/length
    hits_at_5 = hits_at_5/length
    hits_at_10 = hits_at_10/length
    mrr_at_10 = mrr_at_10/length

    return hits_at_1, hits_at_5, hits_at_10, mrr_at_10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset_helper(path):
    data = datasets.load_dataset(
    'json',
    data_files=path,
    ignore_verifications=False,
    cache_dir='cache'
    )['train']

    return data

def get_subsampled_train_dataloader(old_docs_list, new_docs_list, train_data, train_data_new, generated_queries, generated_queries_new, tokenizer, doc_class, batch_size):
    random.shuffle(old_docs_list)
    filter_docs = old_docs_list[:len(new_docs_list)]
    import pdb; pdb.set_trace()
    train_data_tp = train_data.filter(lambda example: example['doc_id'] in filter_docs)
    train_data_new_tp = train_data_new.filter(lambda example: example['doc_id'] in filter_docs)
    generated_queries_tp = generated_queries.filter(lambda example: example['doc_id'] in filter_docs)
    generated_queries_new_tp = generated_queries_new.filter(lambda example: example['doc_id'] in filter_docs)

    DSIQG_train_tp = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data_tp, doc_class=doc_class)
    DSIQG_train_new_tp = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data_new_tp, doc_class=doc_class)
    gen_queries_tp = GenPassageDataset(tokenizer=tokenizer, datadict = generated_queries_tp, doc_class=doc_class)
    gen_queries_new_tp = GenPassageDataset(tokenizer=tokenizer, datadict = generated_queries_new_tp, doc_class=doc_class)

    train_dataset_tp = ConcatDataset([DSIQG_train_tp, DSIQG_train_new_tp, gen_queries_tp, gen_queries_new_tp])

    train_dataloader = DataLoader(train_dataset_tp, 
                                batch_size=batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=True,
                                drop_last=False)

    return train_dataloader

def main():

    args = get_arguments()

    set_seed(args.seed)

    if not args.validate_only or not args.test_only:
        if args.wandb_name:
            wandb.init(project="dsi-string-new", name=args.wandb_name)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )

    logging.basicConfig(filename=f'{args.output_dir}/out.log', encoding='utf-8', level=logging.DEBUG)

    device = torch.device("cuda")

    logging.info(f'Device: {device}')

    train_data = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'trainqueries.json'))
    generated_queries = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'passages_seen.json'))
    val_data = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'valqueries.json'))

    logger.info('train, generated, val loaded')

    if args.doc_index:
        doc_data = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'doc.json'))

    if args.base_data_dir_new or args.test_only:
        train_data_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'trainqueries.json'))
        generated_queries_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'passages_seen.json'))
        val_data_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'valqueries.json'))


        logger.info('new train, generated, val set loaded')

        if args.doc_index:
            doc_data_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'doc.json'))

    if args.test_only:
        test_data = load_dataset_helper(os.path.join(args.base_data_dir, 'old_docs', 'testqueries.json'))
        test_data_new = load_dataset_helper(os.path.join(args.base_data_dir_new, 'testqueries.json'))

        logger.info('test set loaded')

    old_docs_list = joblib.load(os.path.join(args.base_data_dir, 'old_docs', 'doc_list.pkl'))
    class_num = len(old_docs_list)
    if args.new_only:
        doc_class = joblib.load(os.path.join(args.base_data_dir, 'old_docs', 'doc_class.pkl'))
        class_num = len(doc_class)
    if args.base_data_dir_new or args.test_only:
        new_docs_list = joblib.load(os.path.join(args.base_data_dir_new, 'doc_list.pkl'))

        if args.filter_num!=-1:
            filter_docs = new_docs_list[:args.filter_num]
            train_data_new = train_data_new.filter(lambda example: example['doc_id'] in filter_docs)
            generated_queries_new = generated_queries_new.filter(lambda example: example['doc_id'] in filter_docs)
            val_data_new = val_data_new.filter(lambda example: example['doc_id'] in filter_docs)
            if args.doc_index:
                doc_data_new = doc_data_new.filter(lambda example: example['doc_id'] in filter_docs)
            if args.test_only:
                test_data_new = test_data_new.filter(lambda example: example['doc_id'] in filter_docs)
            class_num += args.filter_num
        else:
            class_num += len(new_docs_list)

    logger.info(f'Class number {class_num}')

    logger.info(f'Loading Model and Tokenizer for {args.model_name}')

    if args.model_name == 't5-base':
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir='cache')
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir='cache')
    else:
        print('Model not supported')
        raise NotImplementedError
    
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit() and len(token) == 1:
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit() and len(token) == 1:
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    global id2token_map
    for token_id in INT_TOKEN_IDS:
        id2token_map[token_id] = tokenizer.decode(token_id)
    id2token_map[0] = ""
    id2token_map[1] = ""

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    # load saved model if initialize_model
    # TODO check if anything needs to be changed for string dsi
    if args.initialize_model:
        load_saved_weights(model, args.initialize_model, strict_set=False, load_classifier=False)

    model = torch.nn.DataParallel(model)
    model.to(device)

    logger.info('model loaded')

    doc_class = joblib.load(os.path.join(args.base_data_dir, 'old_docs', 'doc_class.pkl'))

    DSIQG_train = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data, doc_class=doc_class)
    DSIQG_val = DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data, doc_class=doc_class)
    gen_queries = GenPassageDataset(tokenizer=tokenizer, datadict = generated_queries, doc_class=doc_class)

    Train_DSIQG = ConcatDataset([DSIQG_train, gen_queries])
    if args.doc_index:
        DSIQG_doc = DSIqgDocDataset(tokenizer=tokenizer, datadict = doc_data, doc_class=doc_class)
        Train_DSIQG = ConcatDataset([Train_DSIQG, DSIQG_doc])
    Val_DSIQG = DSIQG_val

    if args.base_data_dir_new or args.test_only:
        DSIQG_train_new = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data_new, doc_class=doc_class)
        DSIQG_val_new = DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data_new, doc_class=doc_class)
        gen_queries_new = GenPassageDataset(tokenizer=tokenizer, datadict = generated_queries_new, doc_class=doc_class)

        if args.new_only:
            Train_DSIQG = ConcatDataset([DSIQG_train_new, gen_queries_new])
        else:
            Train_DSIQG = ConcatDataset([Train_DSIQG, DSIQG_train_new, gen_queries_new])
        if args.doc_index:
            DSIQG_doc_new = DSIqgDocDataset(tokenizer=tokenizer, datadict = doc_data_new, doc_class=doc_class)
            Train_DSIQG = ConcatDataset([Train_DSIQG, DSIQG_doc_new])
        Val_DSIQG_NEW = DSIQG_val_new

    if args.test_only:
        DSIQG_test = DSIqgTrainDataset(tokenizer=tokenizer, datadict = test_data, doc_class=doc_class)
        DSIQG_test_new = DSIqgTrainDataset(tokenizer=tokenizer, datadict = test_data_new, doc_class=doc_class)


    def get_parameter_names(model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result
    
    length = len(train_data) + len(generated_queries)

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, **{'lr': args.learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-08})
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.9, end_factor=1, total_iters=10)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=1000000)

    if args.base_data_dir_new or args.test_only:
        if args.new_only:
            length = len(train_data_new) + len(generated_queries_new)
        else:
            length += len(train_data_new) + len(generated_queries_new)
    if args.doc_index:
        length += len(doc_data)

    logger.info(f'dataset size:, {length}')

    val_length = len(val_data)

    logger.info(f'val_ dataset size:, {val_length}')

    if args.base_data_dir_new or args.test_only:
        val_length_new = len(val_data_new)
        logger.info(f'val_ new dataset size:, {val_length_new}')

    if args.test_only:
        test_length = len(test_data)
        test_length_new = len(test_data_new)
        logger.info(f'test dataset size:, {test_length}')
        logger.info(f'test new dataset size:, {test_length_new}')



    # import pdb; pdb.set_trace()

    val_dataloader = DataLoader(Val_DSIQG, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=False,
                                drop_last=False)

    def _get_train_sampler():
        generator = torch.Generator()
        generator.manual_seed(6909045637428952499)
        return RandomSampler(Train_DSIQG, generator=generator)

    
    
    train_sampler = _get_train_sampler()
    
    train_dataloader = DataLoader(Train_DSIQG, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                sampler=train_sampler,
                                drop_last=False)

    if args.base_data_dir_new or args.test_only:
        val_dataloader_new = DataLoader(Val_DSIQG_NEW, 
                                    batch_size=args.batch_size,
                                    collate_fn=IndexingCollator(
                                    tokenizer,
                                    padding='longest'),
                                    shuffle=False,
                                    drop_last=False)

    if args.test_only:
        test_dataloader = DataLoader(DSIQG_test, 
                                    batch_size=args.batch_size,
                                    collate_fn=IndexingCollator(
                                    tokenizer,
                                    padding='longest'),
                                    shuffle=False,
                                    drop_last=False)

        test_dataloader_new = DataLoader(DSIQG_test_new, 
                                    batch_size=args.batch_size,
                                    collate_fn=IndexingCollator(
                                    tokenizer,
                                    padding='longest'),
                                    shuffle=False,
                                    drop_last=False)

    # global_step = 0

    # global_step=0
    # train(args, model, train_dataloader, optimizer, length)
    # import pdb; pdb.set_trace()
    # train(args, model, train_dataloader, optimizer, length, scheduler)
    if args.test_only:
        logger.info('Testing')
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, test_dataloader, restrict_decode_vocab)
        logger.info(f'Test Accuracy: {hit_at_1} / {test_length} = {hit_at_1/test_length}')
        logger.info(f'Test Hits@5: {hit_at_5} / {test_length} = {hit_at_5/test_length}')
        logger.info(f'Test Hits@10: {hit_at_10} / {test_length} = {hit_at_10/test_length}')
        logger.info(f'MRR@10: {mrr_at_10} / {test_length} = {mrr_at_10/test_length}')

        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, test_dataloader_new, restrict_decode_vocab)
        logger.info(f'Test Accuracy: {hit_at_1} / {test_length_new} = {hit_at_1/test_length_new}')
        logger.info(f'Test Hits@5: {hit_at_5} / {test_length_new} = {hit_at_5/test_length_new}')
        logger.info(f'Test Hits@10: {hit_at_10} / {test_length_new} = {hit_at_10/test_length_new}')
        logger.info(f'MRR@10: {mrr_at_10} / {test_length_new} = {mrr_at_10/test_length_new}')

    
    # hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader, restrict_decode_vocab)

    # logger.info(f'Validation accuracy:')
    # logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
    # logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
    # logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
    # logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')

    if args.base_data_dir_new or args.test_only:
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader_new, restrict_decode_vocab)

        logger.info(f'Evaluating on the new dataset')
        logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length_new} = {hit_at_1/val_length_new}')
        logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length_new} = {hit_at_5/val_length_new}')
        logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length_new} = {hit_at_10/val_length_new}')
        logger.info(f'MRR@10: {mrr_at_10} / {val_length_new} = {mrr_at_10/val_length_new}')

    if args.test_only:
        return

    if not args.validate_only:
        # if args.wandb_name:
        #     wandb.log({'Hits@1': hit_at_1/val_length, 'Hits@5': hit_at_5/val_length, \
        #             'Hits@10': hit_at_10/val_length, 'MRR@10': mrr_at_10/val_length, "epoch": 0})
        #     if args.base_data_dir_new:
        #         wandb.log({'Hits@1_new': hit_at_1/val_length_new, 'Hits@5_new': hit_at_5/val_length_new, \
        #                 'Hits@10_new': hit_at_10/val_length_new, 'MRR@10_new': mrr_at_10/val_length_new, "epoch": 0})

        for i in range(args.train_epochs):
            logger.info(f"Epoch: {i+1}")
            print(f"Learning Rate: {scheduler.get_last_lr()}")
            if args.get_subsampled_train_dataloader:
                train_dataloader = get_subsampled_train_dataloader(old_docs_list, new_docs_list, train_data, train_data_new, 
                    generated_queries, generated_queries_new, tokenizer, doc_class, args.batch_size)
            train(args, model, train_dataloader, optimizer, length, scheduler, i)

            # logger.info(f'Train Accuracy: {correct_ratio_train}')
            # logger.info(f'Train Loss: {tr_loss}')

            # scheduler.step()
            hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader, restrict_decode_vocab, i)

            logger.info(f'Epoch: {i+1}')
            logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
            logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
            logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
            logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')

            if args.wandb_name:
                wandb.log({'Hits@1': hit_at_1/val_length, 'Hits@5': hit_at_5/val_length, \
                    'Hits@10': hit_at_10/val_length, 'MRR@10': mrr_at_10/val_length, "epoch": i+1})

            if args.base_data_dir_new:
                hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader_new, restrict_decode_vocab)

                logger.info(f'Evaluating on the new dataset')
                logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length_new} = {hit_at_1/val_length_new}')
                logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length_new} = {hit_at_5/val_length_new}')
                logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length_new} = {hit_at_10/val_length_new}')
                logger.info(f'MRR@10: {mrr_at_10} / {val_length_new} = {mrr_at_10/val_length_new}')

                if args.wandb_name:
                    wandb.log({'Hits@1_new': hit_at_1/val_length_new, 'Hits@5_new': hit_at_5/val_length_new, \
                        'Hits@10_new': hit_at_10/val_length_new, 'MRR@10_new': mrr_at_10/val_length_new, "epoch": i+1})

            cp = save_checkpoint(args, model, i+1, args.output_name)
            logger.info('Saved checkpoint at %s', cp)
                                               

if __name__ == "__main__":
    main()


# 12