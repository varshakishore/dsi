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
    

class GenPassageDataset(Dataset):
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

        input_ids = self.tokenizer(data['gen_question'],
                                   return_tensors="pt",
                                   truncation="only_first",
                                  max_length=32).input_ids[0]
        
        return input_ids, data['doc_id']

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
        default=None,
        type=int,
        required=True,
        help="Number of train epochs",
    )

    parser.add_argument(
        "--model_name",
        default='T5-base',
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
        "--learning_rate",
        default=5e-4,
        type=float,
        required=True,
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
        "--optimized_embeddings",
        default=None,
        type=str,
        help="file for the optimized embedding matrix",
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

    args = parser.parse_args()

    return args



@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)
                
        inputs['labels'] = torch.Tensor(docids).long()
        return inputs

    
def train(args, model, train_dataloader,optimizer, length):

    model.train()
        
    total_correct_predictions = 0
    tr_loss = 0

    device = torch.device('cuda')
    loss_func  = torch.nn.CrossEntropyLoss()

    for i, inputs in enumerate(train_dataloader):

        inputs.to(device)

        decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0],1))

        if args.model_name == 'T5-base':
            logits = model(input_ids=inputs['input_ids'].long(), decoder_input_ids=decoder_input_ids.long()).squeeze()
        elif args.model_name == 'bert-base-uncased':
            logits = model(inputs['input_ids'], inputs['attention_mask'])
        _, docids = torch.max(logits, 1)
        
        loss = loss_func(logits,torch.tensor(inputs['labels']).long())

        correct_cnt = (docids == inputs['labels']).sum()
        
        tr_loss += loss.item()
        
        total_correct_predictions += correct_cnt

        if (i + 1) % args.logging_step == 0:
            logger.info(f'Train step: {i}, loss: {(tr_loss/i)}')

        loss.backward()
        optimizer.step()
        model.zero_grad()
        # global_step += 1

    
    correct_ratio = float(total_correct_predictions / length) 
    

    logger.info(f'Train accuracy:{correct_ratio}')
    logger.info(f'Loss:{tr_loss/(i+1)}')
    return correct_ratio, tr_loss

def validate(args, model, val_dataloader):

    model.eval()

    hit_at_1 = 0
    hit_at_10 = 0
    mrr_at_10 = 0
    hit_at_5 = 0
    val_loss = 0

    device = torch.device('cuda')


    for i,inputs in tqdm(enumerate(val_dataloader), desc='Evaluating dev queries'):
                    
        inputs.to(device)            
        
        with torch.no_grad():
                            
            decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0],1))
            decoder_input_ids = decoder_input_ids.to(device)

            if args.model_name == 'T5-base':
                logits = model(input_ids=inputs['input_ids'].long(), decoder_input_ids=decoder_input_ids.long()).squeeze()
            elif args.model_name == 'bert-base-uncased':
                logits = model(inputs['input_ids'], inputs['attention_mask'])
            

            loss = torch.nn.CrossEntropyLoss()(logits,torch.tensor(inputs['labels']).long())

            val_loss += loss.item()
            
            _, docids = torch.max(logits, 1)
    
            hit_at_1 += (docids == inputs['labels']).sum()

            max_idxs_5 = torch.argsort(logits, 1, descending=True)[:, :5]
            hit_at_5 += (max_idxs_5 == inputs['labels'].unsqueeze(1)).any(1).sum()

            # compute recall@10
            max_idxs_10 = torch.argsort(logits, 1, descending=True)[:, :10]
            hit_at_10 += (max_idxs_10 == inputs['labels'].unsqueeze(1)).any(1).sum()

            #compute mrr@10. Sum will later be divided by number of elements

            mrr_at_10 += (1/ (torch.where(max_idxs_10 == inputs['labels'][:, None])[1] + 1)).sum()

    
    logger.info(f'Validation Loss: {val_loss/(i+1)}')
            
            
    return hit_at_1, hit_at_5, hit_at_10, mrr_at_10

def validate_script(args, new_validation_subset=False, split="new_val"):

    _, model, train_dataloader, val_dataloader, new_train_dataloader, new_val_dataloader, new_extq_dataloader, old_extq_dataloader, class_num_old = getModelDataloader(args, new_validation_subset=new_validation_subset)

    # we are computing validation for doc_ids>=9000
    if new_validation_subset:
        model.module.classifier.weight.data[-714:] = model.module.classifier.weight.data[class_num_old:class_num_old+714]
        model.module.classifier.weight.data[class_num_old:class_num_old+9000] = model.module.classifier.weight.data[:9000]

    if split == "new_val":
        to_eval = new_val_dataloader
    elif split == "old_val":
        to_eval = val_dataloader
    elif split == "new_gen":
        to_eval = new_extq_dataloader
    elif split == "old_gen":
        to_eval = old_extq_dataloader

    hits_at_1, hits_at_5, hits_at_10, mrr_at_10 = validate(args, model, to_eval)
    length = len(to_eval.dataset)
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


def getModelDataloader(args, new_validation_subset=False):
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    device = torch.device("cuda")

    logging.info(f'Device: {device}')

    train_data = datasets.load_dataset(
    'json',
    data_files='/home/vk352/ANCE/NQ320k_dataset_v2/trainqueries_extended.json',
    ignore_verifications=False,
    cache_dir='cache'
    )['train']

    print('train set loaded')

    generated_queries = datasets.load_dataset(
        'json',
        data_files='/home/vk352/ANCE/NQ320k_dataset_v2/passages.json',
        ignore_verifications=False,
        cache_dir='cache'
    )['train']   

    extended_generated_queries = datasets.load_dataset(
        'json',
        data_files='/home/vk352/ANCE/NQ320k_dataset_v2/passages_extended.json',
        ignore_verifications=False,
        cache_dir='cache'
    )['train']

    logger.info('passages loaded')

    val_data = datasets.load_dataset(
    'json',
    data_files='/home/vk352/ANCE/NQ320k_dataset_v2/testqueries.json',
    ignore_verifications=False,
    cache_dir='cache'
    )['train']

    logger.info('test set loaded')

    class_num=int(len(generated_queries)/10)

    # filter and keep the first 100k classes
    train_data_tp = train_data.filter(lambda example: example['doc_id'] <= 100000)
    val_data_tp = val_data.filter(lambda example: example['doc_id'] <= 100000)
    generated_queries_tp = generated_queries.filter(lambda example: example['doc_id'] <= 100000)

    new_train_data = train_data.filter(lambda example: example['doc_id'] > 100000)
    if new_validation_subset:
        new_val_data = val_data.filter(lambda example: example['doc_id'] >= 100000+9000)
    else:
        new_val_data = val_data.filter(lambda example: example['doc_id'] > 100000)
    new_generated_queries = generated_queries.filter(lambda example: example['doc_id'] > 100000)

    ext_gen_queries_old = extended_generated_queries.filter(lambda example: example['doc_id'] <= 100000)
    ext_gen_queries_new = extended_generated_queries.filter(lambda example: example['doc_id'] > 100000)

    train_data = train_data_tp
    val_data = val_data_tp
    generated_queries = generated_queries_tp
    logger.info(f"Filtered set: Train-{len(train_data)}, Test-{len(val_data)}, Generated-{len(generated_queries)}") 
    logger.info(f"Filtered new set: Train-{len(new_train_data)}, Test-{len(new_val_data)}, Generated-{len(new_generated_queries)}") 


    class_num_old = int(len(generated_queries) /10)

    logger.info(f'Class number {class_num}')

    logger.info(f'Loading Model and Tokenizer for {args.model_name}')

    if args.model_name == 'T5-base':
        model = T5Model_projection(class_num)
        tokenizer = T5Tokenizer.from_pretrained('t5-base',cache_dir='cache')
    elif args.model_name == 'bert-base-uncased':
        model = QueryClassifier(class_num)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    # load saved model if initialize_model
    # TODO extend for T5
    if args.initialize_model:
        load_saved_weights(model, args.initialize_model, strict_set=False)


    ### Use pre_calculated weights to initialize the projection matrix
    if args.initialize_embeddings or args.optimized_embeddings:
        if args.optimized_embeddings:
            embedding_matrix = joblib.load(args.optimized_embeddings)
            model.classifier.weight.data = embedding_matrix.detach().to('cpu')
        elif args.ance_embeddings:
            embedding_matrix = load_ance_embeddings(args.initialize_embeddings, 15000)
            # model.classifier.weight.data = torch.from_numpy(embedding_matrix[:class_num_old])
            model.classifier.weight.data[class_num_old:] = torch.from_numpy(embedding_matrix[class_num_old:])
        else:
            embedding_matrix = joblib.load(args.initialize_embeddings)
            model.decoder.projection.weight.data=torch.from_numpy(embedding_matrix.to_numpy())      
        logger.info("weights for projection layer loaded")

    # TODO check what the name in the T5 model is
    if args.freeze_base_model:
        for name, param in model.named_parameters():
            if name !='classifier.weight':
                param.requires_grad = False
            else:
                print("Unfrozen weight:", name)

    model = torch.nn.DataParallel(model)
    model.to(device)

    logger.info('model loaded')

    DSIQG_train = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data)
    DSIQG_val = DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data)
    gen_queries = GenPassageDataset(tokenizer=tokenizer, datadict = generated_queries)
    Train_DSIQG = ConcatDataset([DSIQG_train, gen_queries])
    Val_DSIQG = DSIQG_val

    new_DSIQG_train = DSIqgTrainDataset(tokenizer=tokenizer, datadict = new_train_data)
    new_DSIQG_val = DSIqgTrainDataset(tokenizer=tokenizer, datadict = new_val_data)
    new_gen_queries = GenPassageDataset(tokenizer=tokenizer, datadict = new_generated_queries)
    new_Train_DSIQG = ConcatDataset([new_DSIQG_train, new_gen_queries])
    new_Val_DSIQG = new_DSIQG_val

    new_ext_q = GenPassageDataset(tokenizer=tokenizer, datadict = ext_gen_queries_new)
    old_ext_q = GenPassageDataset(tokenizer=tokenizer, datadict = ext_gen_queries_old)


    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.9, end_factor=1, total_iters=10)

    length = len(train_data) + len(generated_queries)

    logger.info(f'dataset size:, {length}')

    val_length = len(val_data)
    new_val_length = len(new_val_data)

    logger.info(f'val_ dataset size:, {val_length}')
    logger.info(f'new val_ dataset size:, {new_val_length}')


    val_dataloader = DataLoader(Val_DSIQG, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=False,
                                drop_last=False)

    train_dataloader = DataLoader(Train_DSIQG, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=True,
                                drop_last=False)

    new_val_dataloader = DataLoader(new_Val_DSIQG, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=False,
                                drop_last=False)

    new_train_dataloader = DataLoader(new_Train_DSIQG, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=True,
                                drop_last=False)

    new_extq_dataloader = DataLoader(new_ext_q, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=True,
                                drop_last=False)   

    old_extq_dataloader = DataLoader(old_ext_q, 
                                batch_size=args.batch_size,
                                collate_fn=IndexingCollator(
                                tokenizer,
                                padding='longest'),
                                shuffle=True,
                                drop_last=False)                         


    return args, model, train_dataloader, val_dataloader, new_train_dataloader, new_val_dataloader, new_extq_dataloader, old_extq_dataloader, class_num_old

def main():
    args = get_arguments()

    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )

    logging.basicConfig(filename=f'{args.output_dir}/out.log', encoding='utf-8', level=logging.DEBUG)
    
    _, model, train_dataloader, val_dataloader, new_train_dataloader, new_val_dataloader, _, _, class_num_old = getModelDataloader(args)

    # global_step = 0

    # global_step=0
    
    logger.info('Original Validation set')
    hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader)

    val_length = len(val_dataloader.dataset)
    new_val_length = len(new_val_dataloader.dataset)

    logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
    logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
    logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
    logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')
    logger.info(f'Norm: {torch.norm(model.module.classifier.weight.data[:class_num_old], dim=1).mean()}')

    logger.info('New Validation set')
    hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, new_val_dataloader)

    logger.info(f'Evaluation Accuracy: {hit_at_1} / {new_val_length} = {hit_at_1/new_val_length}')
    logger.info(f'Evaluation Hits@5: {hit_at_5} / {new_val_length} = {hit_at_5/new_val_length}')
    logger.info(f'Evaluation Hits@10: {hit_at_10} / {new_val_length} = {hit_at_10/new_val_length}')
    logger.info(f'MRR@10: {mrr_at_10} / {new_val_length} = {mrr_at_10/new_val_length}')
    logger.info(f'Norm: {torch.norm(model.module.classifier.weight.data[class_num_old:], dim=1).mean()}')

    # logger.info('-' * 100)
    # # normalize
    # mean_vector = model.module.classifier.weight.data[:class_num_old].mean(dim=0)
    # model.module.classifier.weight.data[:class_num_old] = model.module.classifier.weight.data[:class_num_old] - mean_vector

    # mean_vector = model.module.classifier.weight.data[class_num_old:].mean(dim=0)
    # model.module.classifier.weight.data[class_num_old:] = model.module.classifier.weight.data[class_num_old:] - mean_vector
    
    # # model.module.classifier.weight.data = torch.nn.functional.normalize(model.module.classifier.weight.data, dim=1)


    # logger.info('with new weights')
    # hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader)

    # logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
    # logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
    # logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
    # logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')
    # logger.info(f'Norm: {torch.norm(model.module.classifier.weight.data[:class_num_old], dim=1).mean()}')

    # logger.info('New Validation set')
    # hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, new_val_dataloader)

    # logger.info(f'Evaluation Accuracy: {hit_at_1} / {new_val_length} = {hit_at_1/new_val_length}')
    # logger.info(f'Evaluation Hits@5: {hit_at_5} / {new_val_length} = {hit_at_5/new_val_length}')
    # logger.info(f'Evaluation Hits@10: {hit_at_10} / {new_val_length} = {hit_at_10/new_val_length}')
    # logger.info(f'MRR@10: {mrr_at_10} / {new_val_length} = {mrr_at_10/new_val_length}')
    # logger.info(f'Norm: {torch.norm(model.module.classifier.weight.data[class_num_old:], dim=1).mean()}')

    

    # if not args.validate_only:
    #     for i in range(args.train_epochs):
    #         logger.info(f"Epoch: {i+1}")
    #         print(f"Learning Rate: {scheduler.get_last_lr()}")
    #         train(args, model, new_train_dataloader, optimizer, length)

    #         # logger.info(f'Train Accuracy: {correct_ratio_train}')
    #         # logger.info(f'Train Loss: {tr_loss}')

    #         scheduler.step()
    #         logger.info(f'Epoch: {i+1}')
    #         logger.info('Original Validation set')
    #         hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, val_dataloader)

    #         logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
    #         logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
    #         logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
    #         logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')

    #         logger.info('New Validation set')
    #         hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(args, model, new_val_dataloader)

    #         logger.info(f'Evaluation Accuracy: {hit_at_1} / {new_val_length} = {hit_at_1/new_val_length}')
    #         logger.info(f'Evaluation Hits@5: {hit_at_5} / {new_val_length} = {hit_at_5/new_val_length}')
    #         logger.info(f'Evaluation Hits@10: {hit_at_10} / {new_val_length} = {hit_at_10/new_val_length}')
    #         logger.info(f'MRR@10: {mrr_at_10} / {new_val_length} = {mrr_at_10/new_val_length}')

    #         cp = save_checkpoint(args,model,i+1)
    #         logger.info('Saved checkpoint at %s', cp)
                                               

if __name__ == "__main__":
    main()


