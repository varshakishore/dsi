import argparse
import joblib
import torch
import numpy
from BertModel import QueryClassifier
from utils import *
import time
from torch.optim import LBFGS, SGD
from tqdm import tqdm
from functools import partial

# ax imports for hyperparameter optimization
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.service.utils.report_utils import exp_to_df

from dsi_model_continual import validate_script

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize(train_q,
                num_qs,
                embeddings_path, 
                model_path,
                multiple_queries=False):
    set_seed()
    sentence_embeddings = joblib.load(embeddings_path)
    class_num = 100001
    model = QueryClassifier(class_num)
    load_saved_weights(model, model_path, strict_set=False)
    classifier_layer = model.classifier.weight.data


    if not multiple_queries:
        embeddings = sentence_embeddings[:1000010][::10]
        embeddings_new = sentence_embeddings[1000010:][::10]
    else:
        embeddings = sentence_embeddings[:1000010][::10]
        num_new_docs = 9714
        embeddings_new = torch.zeros(num_new_docs * num_qs, 768)
        all_embeddings_new = sentence_embeddings[1000010:]
        for i in range(num_new_docs):
            embeddings_new[i * num_qs : (i+1) * num_qs, :] = all_embeddings_new[i*10 : i*10 + num_qs, :]
        
    if train_q:
        print('using train set queries...')
        import pdb; pdb.set_trace()
        train_newqs = joblib.load('/home/cw862/DSI/dsi/train/train_newqs.pkl')
        embeddings_new = torch.cat((embeddings_new, train_newqs))

    return sentence_embeddings, embeddings, embeddings_new, classifier_layer

def addDocs(args, args_valid=None, ax_params=None):
    global time
    global start
    timelist = []
    max_val =[]
    failed_docs = []
    
    _, embeddings, embeddings_new, classifier_layer = initialize(args.train_q, args.num_qs, args.embeddings_path, args.model_path, args.multiple_queries)
    if args.num_new_docs is None:
        num_new_embeddings = len(embeddings_new)
        if args.multiple_queries:
            num_new_docs = num_new_embeddings//args.num_qs
        else:
            num_new_docs = num_new_embeddings
    else:
        num_new_docs = args.num_new_docs

    if ax_params:
        lr = ax_params['lr']; lam = ax_params['lambda']; m1 = ax_params['m1']; m2 = ax_params['m2']
        # num_new_docs = 500
        start_doc = 9000
    else:
        lr = args.lr; lam = args.lam; m1 = args.m1; m2 = args.m2; start_doc = 0

    if args.train_q:
        train_new_ids = joblib.load('/home/cw862/DSI/dsi/train/train_new_ids.pkl')
        train_q_start = args.num_qs * 9714

    added_counter = len(classifier_layer)

    # add rows for the new docs
    classifier_layer = torch.cat((classifier_layer, torch.zeros(num_new_docs, len(classifier_layer[0])).to(classifier_layer.device)))
    embeddings = torch.cat((embeddings, torch.zeros(num_new_docs, len(classifier_layer[0]))))

    step = args.num_qs if args.multiple_queries else 1
    for j in tqdm(range(start_doc*step, num_new_embeddings, step)):
    # for j in range(num_new_docs):
        q = embeddings_new[j]
        if args.init == 'random':
            x = torch.nn.Linear(768, 1).weight.data.squeeze()
        elif args.init == 'mean':
            x = torch.mean(classifier_layer[:added_counter],0).clone().detach()
        elif args.init == 'max':
            x = classifier_layer[torch.argmax(torch.matmul(classifier_layer[:added_counter], q.to(classifier_layer.device))).item()].clone().detach()        
        x = x.to('cuda')
        x.requires_grad = True
        optimizer = SGD([x], lr=lr)
        embeddings = embeddings.to('cuda')
        classifier_layer = classifier_layer.to('cuda')
        q = q.to('cuda')
        max_val = torch.max(torch.matmul(classifier_layer[:added_counter], q))
        if args.multiple_queries:
            qs = embeddings_new[j:j+args.num_qs]
            qs = qs.to('cuda')
            max_vals = [torch.max(torch.matmul(classifier_layer[:added_counter], qs[k])) for k in range(args.num_qs)]

        
        start = time.time()
        for i in range(args.lbfgs_iterations):
            x.requires_grad = True
            def closure():
                if args.multiple_queries:
                    loss = 0
                    for k in range(args.num_qs):
                        loss += lam * max(0, (max_vals[k].item()+m1) - (qs[k].unsqueeze(dim=0) @ x).squeeze())
                    prod = ((x-classifier_layer[:added_counter]) * embeddings[:added_counter]).sum(1) + m2
                    loss += torch.maximum(prod, torch.zeros(len(prod)).to('cuda')).sum()
                else:
                    loss = lam * max(0, (max_val.item()+m1) - (q.unsqueeze(dim=0) @ x).squeeze())
                    prod = ((x-classifier_layer[:added_counter]) * embeddings[:added_counter]).sum(1) + m2
                    loss += torch.maximum(prod, torch.zeros(len(prod)).to('cuda')).sum()

                optimizer.zero_grad()
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            if loss == 0: break
        if loss==0:
            timelist.append(time.time() - start)
        else:
            timelist.append((time.time() - start)*1000)
        if j % 500 == 0:
            print(f'Done {j} in {time.time() - start} seconds; loss={loss}')
        
        if loss != 0: failed_docs.append(j)
                
        # add to classifier_layer and embeddings
        classifier_layer[added_counter] = x
        if args.multiple_queries:
            idx2add = torch.argmax(torch.matmul(qs, x.unsqueeze(dim=1)))
            embeddings[added_counter] = qs[idx2add]
        else:
            embeddings[added_counter] = q
        classifier_layer = classifier_layer.detach()
        embeddings = embeddings.detach()
        loss = loss.detach()
        added_counter += 1

    if ax_params:
        joblib.dump(classifier_layer, os.path.join(args.write_path_dir, 'temp.pkl'))
        hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate_script(args_valid, new_validation_subset=True)
        print(f'Accuracy hits@1: {hit_at_1}')
        print(f'Num failed docs: {len(failed_docs)}')
        print(ax_params)

        return hit_at_1.item()
        
    return failed_docs, classifier_layer, embeddings, np.asarray(timelist).mean(), timelist

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.008, type=float, help="initial learning rate for optimization")
    parser.add_argument("--lam", default=6, type=float, help="lambda for optimization")
    parser.add_argument("--m1", default=0.03, type=float, help="margin for constraint 1")
    parser.add_argument("--m2", default=0.03, type=float, help="margin for constraint 2")
    parser.add_argument("--num_new_docs", default=None, type=int, help="number of new documents to add")
    parser.add_argument("--lbfgs_iterations", default=1000, type=int, help="number of iterations for lbfgs")
    parser.add_argument("--write_path_dir", default=None, type=str, required=True, help="path to write classifier layer to")
    parser.add_argument("--tune_parameters", action="store_true", help="flag for tune parameters")
    parser.add_argument("--multiple_queries", action="store_true", help="flag for multiple_queries")
    parser.add_argument("--num_qs", default=5, type=int, help="number of generated queries to use")
    parser.add_argument("--train_q", action="store_true", help="if we are using train queries to add documents")
    parser.add_argument(
        "--init", 
        default='random', 
        choices=['random', 'mean', 'max'], 
        help='way to initialize the classifier vector')
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

def get_validation_arguments(optimized_embeddings_path):
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

    args = parser.parse_args(['--freeze_base_model', 
    '--output_dir', '/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed_new/', '--model_name', 'bert-base-uncased', 
    '--batch_size', '1600', 
    '--initialize_model', '/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed/projection_nq320k_epoch15',
    '--optimized_embeddings', optimized_embeddings_path])

    return args


def main():
    set_seed()
    args = get_arguments()

    # args_valid = get_validation_arguments(os.path.join(args.write_path_dir, 'temp.pkl'))
    # hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate_script(args_valid, new_validation_subset=True)
    # import pdb; pdb.set_trace()

    if args.tune_parameters:
        print("Tuning parameters")
        os.makedirs(args.write_path_dir, exist_ok=True)
        args_valid = get_validation_arguments(os.path.join(args.write_path_dir, 'temp.pkl'))

        # ax optimize
        best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "lambda", "type": "range", "bounds": [1, 500], "log_scale": True},
            {"name": "m1", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True},
            {"name": "m2", "type": "range", "bounds": [1e-5, 1.0], "log_scale": True},
        ],
        evaluation_function=partial(addDocs, args, args_valid),
        objective_name='time',
        total_trials=30,
        minimize=False,
        )

        print(exp_to_df(experiment)) 
        print(f'best_parameters')
        print(f'lr: {best_parameters["lr"]}')
        print(f'lambda: {best_parameters["lambda"]}')
        print(f'm1: {best_parameters["m1"]}')
        print(f'm2: {best_parameters["m2"]}')

        args.lr = best_parameters['lr']
        args.lam = best_parameters['lambda']
        args.m1 = best_parameters['m1']
        args.m2 = best_parameters['m2']

        if args.write_path_dir is not None:
            print("Writing to directory: ", args.write_path_dir)
            os.makedirs(args.write_path_dir, exist_ok=True)
            with open(os.path.join(args.write_path_dir, 'log.txt'), 'w') as f:
                f.write(f'Best Parameters:\n')
                f.write(f'lr: {args.lr}\n')
                f.write(f'lambda: {args.lam}\n')
                f.write(f'm1: {args.m1}\n')
                f.write(f'm2: {args.m2}\n')
                print('\n')
                f.write(f'experiment: {exp_to_df(experiment).to_csv()}\n')
    
    print("Adding documents")
    failed_docs, classifier_layer, embeddings, avg_time, timelist = addDocs(args)


    if args.write_path_dir is not None:
        print("Writing to directory: ", args.write_path_dir)
        os.makedirs(args.write_path_dir, exist_ok=True)
        joblib.dump(classifier_layer, os.path.join(args.write_path_dir, 'classifier_layer.pkl'))
        joblib.dump(embeddings, os.path.join(args.write_path_dir, 'embeddings.pkl'))
        joblib.dump(failed_docs, os.path.join(args.write_path_dir, 'failed_docs.pkl'))
        joblib.dump(timelist, os.path.join(args.write_path_dir, 'timelist.pkl'))
        with open(os.path.join(args.write_path_dir, 'log.txt'), 'a') as f:
            print('\n')
            f.write(f'Num failed docs: {len(failed_docs)}\n')
            f.write(f'Final time: {np.asarray(timelist).sum()}\n')
            f.write(f'Final time average: {np.asarray(timelist).mean()}\n')

    args_valid = get_validation_arguments(os.path.join(args.write_path_dir, 'classifier_layer.pkl'))
    hit_at_1, hit_at_5, hit_at_10, mrr_at_10 = validate_script(args_valid, new_validation_subset=False)
    if args.write_path_dir is not None:
         with open(os.path.join(args.write_path_dir, 'log.txt'), 'a') as f:
            print('\n')
            print('Validation results on the new test set: \n')
            print(f'hit_at_1: {hit_at_1}')
            print(f'hit_at_5: {hit_at_5}')
            print(f'hit_at_10: {hit_at_10}')
            print(f'mrr_at_10: {mrr_at_10}')

if __name__ == "__main__":
    main()