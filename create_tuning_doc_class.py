import joblib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os
import random
import pickle


def main():
    # load the old document list
    old_docs = joblib.load('/home/vk352/dsi/data/NQ320k/old_docs/doc_list.pkl')
    # load the tuning document list
    tune_docs = joblib.load('/home/vk352/dsi/data/NQ320k/tune_docs/doc_list.pkl')

    tuning_doc2class = {}
    class_num = 0
    for doc_i in old_docs:
        tuning_doc2class[doc_i] = class_num
        class_num += 1
    for doc_i in tune_docs:
        tuning_doc2class[doc_i] = class_num
        class_num += 1

    # Length of the doc2class mapping should be equal to the length of the old docs + the length of the tuning docs
    assert len(tuning_doc2class) == len(old_docs) + len(tune_docs)

    # Mapping for training docs should be the same for tuning and testing
    test_doc2class = joblib.load('/home/vk352/dsi/data/NQ320k/new_docs/doc_class.pkl')
    for doc_i in old_docs:
        assert test_doc2class[doc_i] == tuning_doc2class[doc_i]

    # Save the doc2class mapping for tuning set
    save_dir = '/home/jl3353/dsi/data/NQ320k/tune_docs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    joblib.dump(tuning_doc2class, os.path.join(save_dir, f'doc_class.pkl'))

if __name__ == '__main__':
    main()