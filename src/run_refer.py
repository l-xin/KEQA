# -*- coding: utf-8 -*-

import os

from parse_args import args
from gpt import gpt
from es import es
from refer_build import batch_one_refer, batch_multi_refer, merge_refer

def refer_dataset(datasets, topk=5, max_size=None):
    sub_refer_path_list = []
    for ds in datasets:
        in_path = f"data/{ds}/ref.jsonl"
        refer_path = f"data/{ds}/ref.refer.jsonl"
        sub_refer_path_list.append(refer_path)
        result_path = f"data/{ds}/ref.log.jsonl"
        if ds == "NQ":
            batch_one_refer(in_path, result_path, refer_path, gpt, es, topk, max_size=max_size)
        else:
            yn = ds == "StrategyQA"
            batch_multi_refer(in_path, result_path, refer_path, gpt, es, topk, yn, max_size=max_size)
    
    all_refer_path = args.refer_path
    refer_root = os.path.dirname(all_refer_path)
    if not os.path.exists(refer_root):
        os.mkdir(refer_root)
    merge_refer(sub_refer_path_list, all_refer_path)
    return

if __name__ == "__main__":
    datasets = args.dataset.split(",")
    refer_dataset(datasets, topk=5)
