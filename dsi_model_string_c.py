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
from collections import defaultdict
from utils import *
from lora import *
from transformers import AdamW

import wandb

id2token_map = {}
global_step = 0

class DSIqgTrainDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            datadict,
            doc_class, 
            semantic_id_map=None):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class

        self.semantic_id_map = semantic_id_map
        if semantic_id_map:
            # max length of semantic ids + 1 for eos token
            self.seq_len = len(next(iter(semantic_id_map.values()))) + 1 


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['question'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        if self.semantic_id_map:
            ids = list(map(str, self.semantic_id_map[data['doc_id']]))
            doc_ids = self.tokenizer.encode(ids,
                                       return_tensors="pt",
                                       truncation="only_first",
                                      max_length=self.seq_len, padding='max_length')
            doc_id_str = ' '.join(ids)
        else:
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
            doc_class,
            semantic_id_map=None):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class

        self.semantic_id_map = semantic_id_map
        if semantic_id_map:
            # max length of semantic ids + 1 for eos token
            self.seq_len = len(next(iter(semantic_id_map.values()))) + 1 



    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['doc_text'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        if self.semantic_id_map:
            ids = list(map(str, self.semantic_id_map[data['doc_id']]))
            doc_ids = self.tokenizer.encode(ids,
                                       return_tensors="pt",
                                       truncation="only_first",
                                      max_length=self.seq_len, padding='max_length')
            doc_id_str = ' '.join(ids)
        else:
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
            doc_class,
            semantic_id_map=None):
        super().__init__()
        self.train_data = datadict
        
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class

        self.semantic_id_map = semantic_id_map
        if semantic_id_map:
            # max length of semantic ids + 1 for eos token
            self.seq_len = len(next(iter(semantic_id_map.values()))) + 1 


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        input_ids = self.tokenizer(data['gen_question'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        if self.semantic_id_map:
            ids = list(map(str, self.semantic_id_map[data['doc_id']]))
            doc_ids = self.tokenizer.encode(ids,
                                       return_tensors="pt",
                                       truncation="only_first",
                                      max_length=self.seq_len, padding='max_length')
            doc_id_str = ' '.join(ids)
        else:
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
        "--freeze_lm",
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
        "--lora",
        action="store_true",
        help="whether or not using LORA for continue learning"
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

    parser.add_argument(
        "--semantic_id_path",
        type=str,
        default=None,
        help="path to semantic id map",
    )

    parser.add_argument(
        "--averaging_path",
        type=str,
        default=None,
        help="path to averaging map",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="alpha for averaging",
    )

    parser.add_argument(
        "--ewc",
        action="store_true",
        help="ewc",
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

def train(args, model, train_dataloader, optimizer, length, scheduler=None, i=0, precision_matrices=None, ewc_weights=None):
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
        
        ewc_loss = 0
        if precision_matrices:
            for n, p in model.named_parameters():
                ewc_loss += (precision_matrices[n] * (p - ewc_weights[n]) ** 2).sum()
        
        if (i) % args.logging_step == 0 and i > 0:
            logger.info(f'Train step: {i}, loss: {(tr_loss/args.logging_step).item()}, Global step: {global_step}')
            # logger.info(f'EWC loss: {ewc_loss.item()}, other loss: {loss.item()}')
            if args.wandb_name:
                wandb.log({'train_loss': (tr_loss/args.logging_step).item()})
                wandb.log({'learning_rate': scheduler.get_last_lr()[0]})
                wandb.log({'learning_rate_cp': scheduler.get_last_lr()[0]}, step=global_step)
            
            tr_loss = torch.tensor(0.0).to(device)
        
        loss += 50 * ewc_loss
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

def validate(args, model, val_dataloader, restrict_decode_vocab, atomic_ids=False):

    model.eval()

    hit_at_1 = 0
    hit_at_10 = 0
    mrr_at_10 = 0
    hit_at_5 = 0
    val_loss = 0

    device = torch.device('cuda')


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
            
            if atomic_ids:
                # declare dtype to be string of length i
                seq_length = batch_beams.shape[-1]
                rank_list = np.apply_along_axis(lambda x: np.array(''.join(x), dtype=f'<U{seq_length}'), 1, rank_list)
                rank_list = rank_list.reshape(inputs['input_ids'].shape[0], 10)
            else:
                rank_list_tp = []
                for sublist in rank_list:
                    rank_list_tp.append(' '.join(list(filter(None, sublist))))

                rank_list = np.asarray(rank_list_tp).reshape(inputs['input_ids'].shape[0], 10)

            hit_at_1 += (labels_str == rank_list[:, 0]).sum()
            
            labels_str = np.expand_dims(labels_str, axis=1)

            hit_at_5 += (labels_str == rank_list[:, :5]).any(1).sum()

            hit_at_10 += (labels_str == rank_list).any(1).sum()

            #compute mrr@10. Sum will later be divided by number of elements
            mrr_at_10 += (1/ (np.where(labels_str == rank_list)[1] + 1)).sum()
            
    return hit_at_1, hit_at_5, hit_at_10, mrr_at_10

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

    if not args.validate_only and not args.test_only:
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

    if args.semantic_id_path:
        semantic_id_map = joblib.load(args.semantic_id_path)
        # assuming that the file name is semantic_id_map_i where i is the number of digits
        num_digits = int("/home/vk352/dsi/data/semantic_id_map_30".split('_')[-1])
        atomic_ids = False
    else:
        semantic_id_map = None
        num_digits = 10
        atomic_ids = True
    
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit() and len(token) == 1: 
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            if int(token) < num_digits:
                INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    global id2token_map
    for token_id in INT_TOKEN_IDS:
        id2token_map[token_id] = tokenizer.decode(token_id)
    id2token_map[0] = ""
    id2token_map[1] = ""

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    
    model = torch.nn.DataParallel(model)
    model.to(device)

    if args.initialize_model:
        state_dict = torch.load(args.initialize_model)

        if args.averaging_path and not args.lora:
            state_dict_other = torch.load(args.averaging_path)

            # average the state dicts
            for key in state_dict.keys():
                state_dict[key] = (args.alpha*state_dict[key] + (1-args.alpha)*state_dict_other[key])

        if args.averaging_path and args.lora and args.validate_only:
            lora_statedict = torch.load(args.averaging_path)
            config = LoRAConfig()
            state_dict_other = load_lora_cp(model,lora_statedict,config)
            model.load_state_dict(state_dict_other, strict=False)

            state_dict_other = model.state_dict()

            for key in state_dict.keys():
                state_dict[key] = (args.alpha*state_dict[key] + (1-args.alpha)*state_dict_other[key])

        model.load_state_dict(state_dict, strict=True)

    # training with LoRA
    if args.lora and not args.validate_only:
        # import pdb; pdb.set_trace()
        config = LoRAConfig()
        model = modify_with_lora(model, config)

        print("Trainable parameters")
        print(
            [
                p_name
                for p_name in dict(model.named_parameters()).keys()
                if re.fullmatch(config.trainable_param_names, p_name)
            ]
        )

        def param_name_to_group_name(param_name):
            if False:
                return ".".join(param_name.split(".")[:3])
                # only needed when the model has many trainable parameters, disabled in our expeirments
            else:
                return "."

        param_groups = defaultdict(lambda: {"params": []})
        trainable_param_names = set()
        for (param_name, param) in model.named_parameters():
            if re.fullmatch(config.trainable_param_names, param_name):
                param_groups[param_name_to_group_name(param_name)]["params"].append(param)
                trainable_param_names.add(param_name)
            else:
                param.requires_grad = False

        param_groups = param_groups.values()

        optimizer = AdamW(param_groups, **{'lr': args.learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-08})


    model.to(device)
    # import pdb; pdb.set_trace()

    # freezing the classifier
    if args.freeze_lm:
        for name, param in model.named_parameters():
            if 'shared.weight' in name or 'lm_head.weight' in name:
                param.requires_grad = False

    logger.info('model loaded')

    doc_class = joblib.load(os.path.join(args.base_data_dir, 'old_docs', 'doc_class.pkl'))

    DSIQG_train = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data, doc_class=doc_class, semantic_id_map=semantic_id_map)
    DSIQG_val = DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data, doc_class=doc_class, semantic_id_map=semantic_id_map)
    gen_queries = GenPassageDataset(tokenizer=tokenizer, datadict = generated_queries, doc_class=doc_class, semantic_id_map=semantic_id_map)

    Train_DSIQG = ConcatDataset([DSIQG_train, gen_queries])
    if args.doc_index:
        DSIQG_doc = DSIqgDocDataset(tokenizer=tokenizer, datadict = doc_data, doc_class=doc_class, semantic_id_map=semantic_id_map)
        Train_DSIQG = ConcatDataset([Train_DSIQG, DSIQG_doc])
    Val_DSIQG = DSIQG_val

    if args.base_data_dir_new or args.test_only:
        DSIQG_train_new = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data_new, doc_class=doc_class, semantic_id_map=semantic_id_map)
        DSIQG_val_new = DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data_new, doc_class=doc_class, semantic_id_map=semantic_id_map)
        gen_queries_new = GenPassageDataset(tokenizer=tokenizer, datadict = generated_queries_new, doc_class=doc_class, semantic_id_map=semantic_id_map)

        if args.new_only:
            Train_DSIQG = ConcatDataset([DSIQG_train_new, gen_queries_new])
            if args.ewc:
                Train_DSIQG_ewc = Train_DSIQG
        else:
            Train_DSIQG = ConcatDataset([Train_DSIQG, DSIQG_train_new, gen_queries_new])
        if args.doc_index:
            DSIQG_doc_new = DSIqgDocDataset(tokenizer=tokenizer, datadict = doc_data_new, doc_class=doc_class, semantic_id_map=semantic_id_map)
            Train_DSIQG = ConcatDataset([Train_DSIQG, DSIQG_doc_new])
        Val_DSIQG_NEW = DSIQG_val_new

    if args.test_only:
        DSIQG_test = DSIqgTrainDataset(tokenizer=tokenizer, datadict = test_data, doc_class=doc_class, semantic_id_map=semantic_id_map)
        DSIQG_test_new = DSIqgTrainDataset(tokenizer=tokenizer, datadict = test_data_new, doc_class=doc_class, semantic_id_map=semantic_id_map)


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

    # if not args.lora:
    optimizer = AdamW(optimizer_grouped_parameters, **{'lr': args.learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-08})
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.9, end_factor=1, total_iters=10)
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
    
    if args.ewc:
        train_dataloader_ewc = DataLoader(Train_DSIQG_ewc, 
                                    batch_size=args.batch_size,
                                    collate_fn=IndexingCollator(
                                    tokenizer,
                                    padding='longest'),
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

    if args.ewc:
        precision_matrices = {}
        
        model_ewc_sd = torch.load(args.initialize_model)
        model_ewc = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir='cache')
        model_ewc = torch.nn.DataParallel(model_ewc)
        model_ewc.load_state_dict(model_ewc_sd)
        len_data_ewc = len(train_dataloader_ewc.dataset)
        model_ewc.to(device)

        for n, p in model_ewc.named_parameters():
            precision_matrices[n] = p.clone().detach().fill_(0)

        for inputs in tqdm(train_dataloader_ewc):
            model.zero_grad()
            del inputs['docid_str']
            inputs.to(device)

            loss = model_ewc(input_ids=inputs['input_ids'].long(), attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
            loss.backward()

            for n, p in model_ewc.named_parameters():
                precision_matrices[n].data += p.grad ** 2 / len_data_ewc

        del model_ewc
    
    # global_step = 0

    # global_step=0
    # train(args, model, train_dataloader, optimizer, length)
    # import pdb; pdb.set_trace()
    # train(args, model, train_dataloader, optimizer, length, scheduler)
    if args.test_only:
        logger.info('Testing')
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, test_dataloader, restrict_decode_vocab, atomic_ids)
        logger.info(f'Test Accuracy: {hit_at_1} / {test_length} = {hit_at_1/test_length}')
        logger.info(f'Test Hits@5: {hit_at_5} / {test_length} = {hit_at_5/test_length}')
        logger.info(f'Test Hits@10: {hit_at_10} / {test_length} = {hit_at_10/test_length}')
        logger.info(f'MRR@10: {mrr_at_10} / {test_length} = {mrr_at_10/test_length}')

        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, test_dataloader_new, restrict_decode_vocab, atomic_ids)
        logger.info(f'Test Accuracy: {hit_at_1} / {test_length_new} = {hit_at_1/test_length_new}')
        logger.info(f'Test Hits@5: {hit_at_5} / {test_length_new} = {hit_at_5/test_length_new}')
        logger.info(f'Test Hits@10: {hit_at_10} / {test_length_new} = {hit_at_10/test_length_new}')
        logger.info(f'MRR@10: {mrr_at_10} / {test_length_new} = {mrr_at_10/test_length_new}')

    # import pdb;pdb.set_trace()
    
    hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader, restrict_decode_vocab, atomic_ids)

    logger.info(f'Validation accuracy:')
    logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
    logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
    logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
    logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')

    if args.base_data_dir_new or args.test_only:
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader_new, restrict_decode_vocab, atomic_ids)

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
            if args.ewc:
                train(args, model, train_dataloader, optimizer, length, scheduler, i, precision_matrices, model_ewc_sd)
            else:
                train(args, model, train_dataloader, optimizer, length, scheduler, i)

            # logger.info(f'Train Accuracy: {correct_ratio_train}')
            # logger.info(f'Train Loss: {tr_loss}')

            # scheduler.step()
            # hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader, restrict_decode_vocab, atomic_ids)

            logger.info(f'Epoch: {i+1}')
            # logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
            # logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
            # logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
            # logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')

            # if args.wandb_name:
            #     wandb.log({'Hits@1': hit_at_1/val_length, 'Hits@5': hit_at_5/val_length, \
            #         'Hits@10': hit_at_10/val_length, 'MRR@10': mrr_at_10/val_length, "epoch": i+1})

            if args.base_data_dir_new:
                hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader_new, restrict_decode_vocab, atomic_ids)

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