import argparse
import joblib
import torch
import numpy
from BertModel import QueryClassifier
from utils import *
import time
from torch.optim import LBFGS, SGD
from tqdm import tqdm

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize(embeddings_path, 
                model_path,
                single_embedding=True):
    sentence_embeddings = joblib.load(embeddings_path)
    class_num = 100001
    model = QueryClassifier(class_num)
    load_saved_weights(model, model_path, strict_set=False)
    classifier_layer = model.classifier.weight.data

    if single_embedding:
        embeddings = sentence_embeddings[:1000010][::10]
        embeddings_new = sentence_embeddings[1000010:][::10]
    else:
        embeddings = sentence_embeddings[:1000010]
        embeddings_new = sentence_embeddings[1000010:]
    
    return sentence_embeddings, embeddings, embeddings_new, classifier_layer

def addDocs(args):
    global time
    global start
    timelist = []
    max_val =[]
    failed_docs = []
    
    _, embeddings, embeddings_new, classifier_layer = initialize()
    if args.num_new_docs is None:
        args.num_new_docs = len(embeddings_new)
    
    for j in range(args.num_new_docs):
        q = embeddings_new[j]
        x = torch.nn.Linear(768, 1).weight.data.squeeze()
        x = x.to('cuda')
        x.requires_grad = True
        optimizer = SGD([x], lr=args.lr)
        embeddings = embeddings.to('cuda')
        classifier_layer = classifier_layer.to('cuda')
        q = q.to('cuda')
        max_val = torch.max(torch.matmul(classifier_layer, q))
    #     print("max val: ", max_val)
        
        start = time.time()
        for i in range(args.lbfgs_iterations):
            x.requires_grad = True
            def closure():
                loss = args.lam * max(0, (max_val.item()+args.m1) - (q.unsqueeze(dim=0) @ x).squeeze())
                prod = ((x-classifier_layer) * embeddings).sum(1) + args.m2
                loss += torch.maximum(prod, torch.zeros(len(prod)).to('cuda')).sum()

                optimizer.zero_grad()
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if loss == 0: break
        if j % 100 == 0:
            print(f'Done {j} in {time.time() - start} seconds; loss={loss}')
            
        if loss==0:
            timelist.append(time.time() - start)
                
            # add to classifier_layer and embeddings
            classifier_layer = torch.cat((classifier_layer, x.unsqueeze(dim=0))).float()
            embeddings = torch.cat((embeddings, q.unsqueeze(dim=0)))
        else:
            failed_docs.append(j)

    if args.write_path is not None:
        joblib.dump(classifier_layer, args.write_path)
        
    return failed_docs, classifier_layer, np.asarray(timelist).mean()

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate for optimization")
    parser.add_argument("--lam", default=1, type=float, help="lambda for optimization")
    parser.add_argument("--m1", default=0.05, type=float, help="margin for constraint 1")
    parser.add_argument("--m2", default=0.0005, type=float, help="margin for constraint 2")
    parser.add_argument("--num_new_docs", default=None, type=int, help="number of new documents to add")
    parser.add_argument("--lbfgs_iterations", default=1000, type=int, help="number of iterations for lbfgs")
    parser.add_argument("--write_path", default=None, type=str, help="path to write classifier layer to")
    parser.add_argument(
        "--embeddings_path", 
        default='/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed/nq320k_gen_passage_embeddings.pkl', 
        type=str, 
        help="path to embeddings")
    parser.add_argument(
        "--model_path", 
        default="/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed/projection_nq320k_epoch15", 
        type=str, 
        help="path to model")
    
    args = parser.parse_args()

    return args


def main():
    set_seed()
    args = get_arguments()

    failed_docs, classifier_layer, avg_time = addDocs(args)

if __name__ == "__main__":
    main()