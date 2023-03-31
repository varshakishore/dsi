import joblib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os
import random
import pickle


def main():
    # Save the doc2class mapping for tuning set
    save_dir = '/home/jl3353/dsi/data/NQ320k/new_docs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50]:
        # load the old document list
        old_docs = joblib.load('/home/vk352/dsi/data/NQ320k/old_docs/doc_list.pkl')
        # load the new document list
        new_docs = joblib.load('/home/vk352/dsi/data/NQ320k/new_docs/doc_list.pkl')
        np.random.seed(seed)
        np.random.shuffle(new_docs)
        print(f'Saving {os.path.join(save_dir, f"doc_list_seed{seed}.pkl")}')
        joblib.dump(new_docs, os.path.join(save_dir, f'doc_list_seed{seed}.pkl'))

        new_doc2class = {}
        class_num = 0
        for doc_i in old_docs:
            new_doc2class[doc_i] = class_num
            class_num += 1
        for doc_i in new_docs:
            new_doc2class[doc_i] = class_num
            class_num += 1

        # Length of the doc2class mapping should be equal to the length of the old docs + the length of the tuning docs
        assert len(new_doc2class) == len(old_docs) + len(new_docs)

        # Mapping for training docs should be the same for all test sets
        test_doc2class = joblib.load('/home/vk352/dsi/data/NQ320k/new_docs/doc_class.pkl')
        for doc_i in old_docs:
            assert test_doc2class[doc_i] == new_doc2class[doc_i]
        print(f'Saving {os.path.join(save_dir, f"doc_class_seed{seed}.pkl")}')
        joblib.dump(new_doc2class, os.path.join(save_dir, f'doc_class_seed{seed}.pkl'))
        print(f'Done with seed {seed}')

if __name__ == '__main__':
    main()