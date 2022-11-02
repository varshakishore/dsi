import collections
import os 
import pickle as pkl
import numpy as np
import pandas as pd
from torch import nn
import torch

CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])

def load_ance_embeddings(
        load_path,
        step_num=0):


    data_list = []
    data_list_id = []
    world_size = 4
    prefix_passage = "ann_NQ_test/ann_data/passage_"+ str(step_num)+ "__emb_p_"
    prefix_passage_id = "ann_NQ_test/ann_data/passage_"+ str(step_num)+ "__embid_p_"

    # import pdb; pdb.set_trace()
    for i in range(world_size):
        pickle_path = os.path.join(
            load_path,
            "{1}_data_obj_{0}.pb".format(
                str(i),
                prefix_passage))
        pickle_path_id = os.path.join(
            load_path,
            "{1}_data_obj_{0}.pb".format(
                str(i),
                prefix_passage_id))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pkl.load(handle)
                data_list.append(b)
            with open(pickle_path_id, 'rb') as handle:
                b = pkl.load(handle)
                data_list_id.append(b)
        except BaseException:
            print("No file to load. Check path to embeddings.")
            continue

    data_array_agg = np.concatenate(data_list, axis=0)
    data_array_agg_id = np.concatenate(data_list_id, axis=0)
    
    embedding_matrix = data_array_agg[np.argsort(data_array_agg_id)] 

    # HARDCODING path from NQ v2 for now
    ques2doc = pkl.load(open(os.path.join("/home/vk352/ANCE/NQ320k_dataset_v2", "quesid2docid.pkl"), 'rb'))
    pid2offset, offset2pid = load_mapping(load_path, "pid2offset")
    doc_ids = [ques2doc[offset2pid[i]] for i in range(len(data_array_agg))]

    embedding_matrix_pd = pd.DataFrame(embedding_matrix)
    embedding_matrix_pd.insert(0, "doc_ids", doc_ids)
    # average embeddigns corresponding to the same doc_id
    embedding_matrix_avg = embedding_matrix_pd.groupby('doc_ids').mean()
    assert (embedding_matrix_avg.index.values == [i for i in range(len(embedding_matrix_avg.index.values))]).all()

    return embedding_matrix_avg.to_numpy()

def load_mapping(data_dir, out_name):
    out_path = os.path.join(
        data_dir,
        out_name ,
    )
    pid2offset = {}
    offset2pid = {}
    with open(out_path, 'r') as f:
        for line in f.readlines():
            line_arr = line.split('\t')
            pid2offset[int(line_arr[0])] = int(line_arr[1])
            offset2pid[int(line_arr[1])] = int(line_arr[0])
    return pid2offset, offset2pid

def load_saved_weights(model, model_path, strict_set=False, load_classifier=True):
    state_dict = torch.load(model_path)
    if len(state_dict.keys()) != 6 and load_classifier:
        # model_to_load = get_model_obj(model)
        model = torch.nn.DataParallel(model)
        del state_dict['module.classifier.weight']
        model.load_state_dict(state_dict, strict=strict_set)
        model = model.module
        # TODO remove this temporary hack
        state_dict = torch.load(model_path)
        model.classifier.weight.data[:len(state_dict['module.classifier.weight'])] = state_dict['module.classifier.weight']
    elif len(state_dict.keys()) != 6 and not load_classifier:
        model = torch.nn.DataParallel(model)
        del state_dict['module.classifier.weight']
        model.load_state_dict(state_dict, strict=strict_set)
        model = model.module
    else:
        saved_state = CheckpointState(**state_dict)
        model_to_load = get_model_obj(model)
        model_to_load.load_state_dict(saved_state.model_dict, strict=strict_set)

def save_checkpoint(args, model, epoch) -> str:
    cp = os.path.join(args.output_dir, 'projection_nq320k_epoch' + str(epoch))

    torch.save(model.state_dict(), cp)
    return cp

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model