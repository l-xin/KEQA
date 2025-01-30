# -*- coding: utf-8 -*-

import traceback

from metrics import cal_f1_score
from qa import quiz, quiz_cot, rag, rag_cot
from utils import read_jsonl, save_jsonl

def one_refer(query, true_ans, true_alias, llm, retriever, topk, yn=False, id=None):
    docs = retriever(query=query, topk=topk, id=f"refer-{id}")
    corpus_label = []
    results = []
    quiz_ans, quiz_raw_ans = quiz(query, llm, yn=yn, id=f"refer-quiz-{id}")
    quiz_score = cal_f1_score(quiz_ans, true_ans, true_alias)
    result_item = {"answer": quiz_ans}
    if quiz_ans != quiz_raw_ans:
        result_item["raw_answer"] = quiz_raw_ans
    result_item["score"] = quiz_score
    results.append(result_item)
    for doc_idx, doc in enumerate(docs):
        content = doc["text"]
        rag_ans, rag_raw_ans = rag(query, [content], llm, yn=yn, id=f"refer-rag-{doc_idx+1}-{id}")
        rag_score = cal_f1_score(rag_ans, true_ans, true_alias)
        if rag_score > quiz_score:
            label = 1
        elif rag_score < quiz_score:
            label = -1
        else:
            label = 0
        corpus_label.append({"text": content, "label": label})
        result_item = {"answer": rag_ans}
        if rag_ans != rag_raw_ans:
            result_item["raw_answer"] = rag_raw_ans
        result_item["score"] = rag_score
        result_item["label"] = label
        results.append(result_item)
    return corpus_label, results

def multi_refer(query, true_ans, true_alias, llm, retriever, topk, yn=False, id=None):
    docs = retriever(query=query, topk=topk, id=f"refer-{id}")
    corpus_label = []
    results = []
    pos_docs = []
    quiz_ans, quiz_raw_ans = quiz_cot(query, llm, yn=yn, id=f"refer-quiz-{id}")
    quiz_score = cal_f1_score(quiz_ans, true_ans, true_alias)
    result_item = {"answer": quiz_ans}
    if quiz_ans != quiz_raw_ans:
        result_item["raw_answer"] = quiz_raw_ans
    result_item["score"] = quiz_score
    results.append(result_item)
    for doc_idx, doc in enumerate(docs):
        content = doc["text"]
        rag_ans, rag_raw_ans = rag_cot(query, pos_docs + [content], llm, yn=yn, id=f"refer-rag-{doc_idx+1}-{id}")
        rag_score = cal_f1_score(rag_ans, true_ans, true_alias)
        if rag_score > quiz_score:
            label = 1
            quiz_score = rag_score
            pos_docs.append(content)
        elif rag_score < quiz_score:
            label = -1
        else:
            label = 0
        corpus_label.append({"text": content, "label": label})
        result_item = {"answer": rag_ans}
        if rag_ans != rag_raw_ans:
            result_item["raw_answer"] = rag_raw_ans
        result_item["score"] = rag_score
        result_item["label"] = label
        results.append(result_item)
    return corpus_label, results

def batch_refer(refer_func, in_path, result_path, refer_path, llm, retriever, topk, yn=False, max_size=None):
    data = read_jsonl(in_path, max_size=max_size)
    result_data = []
    refer_data = []
    try:
        for item in data:
            query = item["question"]
            true_ans = item["answer"]
            true_alias = item.get("alias", None)
            corpus_label_item, result_item = refer_func(query, true_ans, true_alias, llm, retriever, topk, yn=yn, id=item["id"])
            refer_data.append({"id": item["id"], "question": item["question"], "pool": corpus_label_item})
            result_data.append({"id": item["id"], "question": item["question"], "results": result_item})
    except:
        save_jsonl(refer_data, refer_path)
        save_jsonl(result_data, result_path)
        traceback.print_exc()
        exit()
    save_jsonl(refer_data, refer_path)
    save_jsonl(result_data, result_path)
    return

def batch_one_refer(in_path, result_path, refer_path, llm, retriever, topk, yn=False, max_size=None):
    batch_refer(one_refer, in_path, result_path, refer_path, llm, retriever, topk, yn=yn, max_size=max_size)
    return

def batch_multi_refer(in_path, result_path, refer_path, llm, retriever, topk, yn=False, max_size=None):
    batch_refer(multi_refer, in_path, result_path, refer_path, llm, retriever, topk, yn=yn, max_size=max_size)
    return

def merge_refer(in_path_list, out_path):
    refer_data = []
    for in_path in in_path_list:
        in_data = read_jsonl(in_path)
        for item in in_data:
            pool = [refer_item for refer_item in item["pool"] if refer_item["label"] != 0]
            if len(pool) > 0:
                item["pool"] = pool
                refer_data.append(item)
    save_jsonl(refer_data, out_path)
    return
