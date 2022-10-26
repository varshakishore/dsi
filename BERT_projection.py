import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from transformers import BertTokenizer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from dataclasses import dataclass
# from T5_base_projection import T5Model_projection
# from T5_base_projection_decoder import T5decoder_projection
from BertModel import QueryClassifier
import random
import logging
logger = logging.getLogger(__name__)
import argparse
import os
import joblib


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
    

class passagesloader(Dataset):
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
        "--initialize",
        default=False,
        type=bool,
        help="whether or not to initialize the projection layer with pre-calculated embedding matrix",
    )

    parser.add_argument(
        "--embedding",
        type=str,
        help="file for the embedding matrix",
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

def validate(model, val_dataloader):

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

def _save_checkpoint(args, model, epoch) -> str:
    cp = os.path.join(args.output_dir, 'projection_nq320k_epoch' + str(epoch))

    torch.save(model.state_dict(), cp)

    logger.info('Saved checkpoint at %s', cp)
    return cp
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    args = get_arguments()

    set_seed()

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

    logger.info('passages loaded')

    val_data = datasets.load_dataset(
    'json',
    data_files='/home/vk352/ANCE/NQ320k_dataset_v2/testqueries.json',
    ignore_verifications=False,
    cache_dir='cache'
    )['train']

    logger.info('test set')

    length_queries = len(generated_queries)
    logger.info('length')
    logger.info(length_queries)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='cache')

    gen_queries = passagesloader(tokenizer=tokenizer, datadict = generated_queries )

    class_num=int(length_queries/10)

    logger.info(f'Class number {class_num}')

    print('Loading model')

    model = QueryClassifier(class_num)


    ### Use pre_calculated weights to initialize the projection matrix
    if args.initialize:
        # import pdb; pdb.set_trace()
        embedding_matrix = joblib.load(args.embedding)
        model.decoder.projection.weight.data=torch.from_numpy(embedding_matrix.to_numpy())      
        logger.info("weights for projection layer loaded")


    model = torch.nn.DataParallel(model)
    model.to(device)

    logger.info('model loaded')

    #### For reproducibility

    # import random
    # import transformers
    # transformers.set_seed(12)
    # torch.manual_seed(12)
    # random.seed(12)

    ######

    DSIQG_train = DSIqgTrainDataset(tokenizer=tokenizer, datadict = train_data)
    DSIQG_val = DSIqgTrainDataset(tokenizer=tokenizer, datadict = val_data)
    gen_queries = passagesloader(tokenizer=tokenizer, datadict = generated_queries )
    Train_DSIQG = ConcatDataset([DSIQG_train, gen_queries])
    Val_DSIQG = DSIQG_val

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.9, end_factor=1, total_iters=10)

    length = len(train_data) + len(generated_queries)

    logger.info(f'dataset size:, {length}')

    val_length = len(val_data)

    logger.info(f'val_ dataset size:, {val_length}')



    # import pdb; pdb.set_trace()

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

    # global_step = 0

    # global_step=0
    



    if not args.validate_only:
        for i in range(args.train_epochs):
            logger.info(f"Epoch: {i+1}")
            print(f"Learning Rate: {scheduler.get_last_lr()}")
            ###### TODO REMOVE HARDCODE
            train(args,model, train_dataloader,optimizer,length)

            # logger.info(f'Train Accuracy: {correct_ratio_train}')
            # logger.info(f'Train Loss: {tr_loss}')

            scheduler.step()
            hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate(model,val_dataloader)

            logger.info(f'Epoch: {i+1}')
            logger.info(f'Evaluation Accuracy: {hit_at_1} / {val_length} = {hit_at_1/val_length}')
            logger.info(f'Evaluation Hits@5: {hit_at_5} / {val_length} = {hit_at_5/val_length}')
            logger.info(f'Evaluation Hits@10: {hit_at_10} / {val_length} = {hit_at_10/val_length}')
            logger.info(f'MRR@10: {mrr_at_10} / {val_length} = {mrr_at_10/val_length}')

            _save_checkpoint(args,model,i+1)
                                               

if __name__ == "__main__":
    main()


