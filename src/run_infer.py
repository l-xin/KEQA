# -*- coding: utf-8 -*-

import os

from parse_args import args
from gpt import gpt
from llama import llama
from es import es
from refer_search import refer_retriever
from decomp import batch_decompose
from infer import batch_infer
from metrics import batch_eval, batch_eval_yn

def decomp_dataset(datasets, max_size=None):
    for ds in datasets:
        in_path = f"data/{ds}/test.jsonl"
        out_path = f"data/{ds}/test.decomp.jsonl"
        if ds != "NQ":
            batch_decompose(in_path, out_path, gpt, max_size=max_size)
    return

def infer_dataset(datasets, sc_num, sc_threshold, n_doc, n_refer, min_refer_label_num, max_size=None):
    # min_refer_label_num: min number of refer with positive & negative label, avoid all positive refer or all negative refer
    for ds in datasets:
        in_path = f"data/{ds}/test.jsonl"
        decomp_path = f"data/{ds}/test.decomp.jsonl"
        if not os.path.exists(decomp_path):
            decomp_path = in_path
        pred_path = f"data/{ds}/test.infer.jsonl"
        eval_path = f"data/{ds}/test.eval.jsonl"
        yn = ds == "StrategyQA"
        batch_infer(decomp_path, pred_path, gpt, es, llama, refer_retriever, sc_num, sc_threshold, n_doc, n_refer, min_refer_label_num, yn, max_size=max_size)
        if yn:
            eval_result =  batch_eval_yn(in_path, pred_path, eval_path, max_size=max_size)
        else:
            eval_result = batch_eval(in_path, pred_path, eval_path, max_size=max_size)
        print(ds, eval_result)
    return

if __name__ == "__main__":
    datasets = args.dataset.split(",")
    decomp_dataset(datasets)
    infer_dataset(datasets, sc_num=args.sc_num, sc_threshold=args.sc_threshold, n_doc=args.doc_num, n_refer=args.refer_num, min_refer_label_num=args.refer_min)
