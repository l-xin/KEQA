# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import os

def get_args():
    parser = ArgumentParser()

    parser.add_argument("--es-host", type=str, default="http://localhost:9200")
    parser.add_argument("--es-index", type=str, default="psgs_w100")
    parser.add_argument("--es-corpus", type=str, default="data/corpus/psgs_w100.tsv")
    parser.add_argument("--es-log", type=str, default="es.log")

    parser.add_argument("--gpt-key", type=str, default=None)
    parser.add_argument("--gpt-url", type=str, default=None)
    parser.add_argument("--gpt-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--gpt-log", type=str, default="gpt.log")

    parser.add_argument("--llama-model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--llama-log", type=str, default="llama.log")
    
    parser.add_argument("--refer-model", type=str, default="bert-base-uncased")
    parser.add_argument("--refer-path", type=str, default="data/refer/refer.jsonl")
    parser.add_argument("--refer-index", type=str, default="data/refer/index.faiss")
    parser.add_argument("--refer-log", type=str, default="refer.log")

    parser.add_argument("--dataset", type=str, default="NQ,StrategyQA,HotpotQA,2WikiMultihopQA")
    parser.add_argument("--sc-num", type=int, default=5)
    parser.add_argument("--sc-threshold", type=int, default=4)
    parser.add_argument("--doc-num", type=int, default=10)
    parser.add_argument("--refer-num", type=int, default=8)
    parser.add_argument("--refer-min", type=int, default=2)
    
    parser.add_argument("--cuda", type=str, default=None)
    
    args = parser.parse_args()
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    return args

args = get_args()
