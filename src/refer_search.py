# -*- coding: utf-8 -*-

import json
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from parse_args import args
from utils import get_logger, read_jsonl

class ReferRetriever:
    def __init__(self):
        self.vector_dim = 768
        self.batch_size = 32
        self.refer_path = args.refer_path
        self.index_path = args.refer_index
        self.model = args.refer_model
        self.device = "cuda" if args.cuda is not None and torch.cuda.is_available() else "cpu"
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.refer_data = []

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModel.from_pretrained(self.model).eval().to(self.device)
        self.log = get_logger("ReferRetriever", args.refer_log)

        self.load_refer_data()
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.log.info(f"building index from {self.index_path} ...")
            self.build_index()
            self.dump_index()
        return
    
    def __call__(self, query, docs, topk, min_label_num, id=None):
        # min_label_num: min number of refer with each label, avoid all positive refer or all negative refer
        hiddens = self.encode_query([query]*len(docs), docs)
        scores, indexes = self.index.search(hiddens, 2 * topk)
        scores, indexes = scores.tolist(), indexes.tolist()
        datas = [[self.refer_data[q_index] for q_index in q_indexes] for q_indexes in indexes]
        assert len(docs) == len(datas) and len(docs) == len(scores)
        results = []
        for doc, score_list, data_list in zip(docs, scores, datas):
            assert len(score_list) == len(data_list)
            label_2_refer = dict()
            for score, data_item in zip(score_list, data_list):
                p_ques, p_doc, p_label = data_item
                refer_item = {"question": p_ques, "text": p_doc, "label": p_label, "score": score}
                label_2_refer.setdefault(p_label, []).append(refer_item)
            for refer_list in label_2_refer.values():
                refer_list.sort(key=lambda refer_item: refer_item["score"], reverse=True)
            result = []
            if 1 in label_2_refer.keys():
                result.extend(label_2_refer[1][:min_label_num])
                label_2_refer[1] = label_2_refer[1][min_label_num:]
            if -1 in label_2_refer.keys():
                result.extend(label_2_refer[-1][:min_label_num])
                label_2_refer[-1] = label_2_refer[-1][min_label_num:]
            rest_refer_list = [refer_item for refer_list in label_2_refer.values() for refer_item in refer_list]
            rest_refer_list.sort(key=lambda refer_item: refer_item["score"], reverse=True)
            result.extend(rest_refer_list[:topk-len(result)])
            result.sort(key=lambda refer_item: refer_item["score"], reverse=True)
            results.append({"doc": doc, "refer": result})
        self.log.info(json.dumps({"id": id, "topk": topk, "min_label_num": min_label_num, "query": query, "results": results}, ensure_ascii=False))
        return results
    
    def load_refer_data(self):
        data = read_jsonl(self.refer_path)
        self.refer_data = []
        for item in data:
            question = item["question"]
            for refer_item in item["pool"]:
                doc = refer_item["text"]
                label = refer_item["label"]
                self.refer_data.append((question, doc, label))
        return
    
    def build_index(self):
        batch_data = []
        batch = []
        for idx, data_item in enumerate(self.refer_data):
            batch.append(data_item)
            if (idx+1) % self.batch_size == 0:
                batch_data.append(batch)
                batch = []
        if len(batch) > 0:
            batch_data.append(batch)
        all_hiddens = None
        for batch in batch_data:
            queries, docs, labels = list(zip(*batch))
            hiddens = self.encode_query(queries, docs)
            if all_hiddens is None:
                all_hiddens = hiddens
            else:
                all_hiddens = np.vstack((all_hiddens, hiddens))
        assert all_hiddens.shape[0] == len(self.refer_data)
        if not self.index.is_trained:
            self.index.train(all_hiddens)
        self.index.add(all_hiddens)
        return
    
    def encode_query(self, queries, docs):
        inputs = [f"Question: {query}\n\nPassage: {doc}" for query, doc in zip(queries, docs)]
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs)["last_hidden_state"][:,0]
        hidden = hidden.cpu().numpy()
        return hidden
    
    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        return
    
    def dump_index(self):
        faiss.write_index(self.index, self.index_path)
        return

refer_retriever = ReferRetriever()
