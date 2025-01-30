# -*- coding: utf-8 -*-

from collections import Counter

from utils import normalize, read_jsonl, save_jsonl

def em_score(pred_ans, true_ans):
    return float(normalize(pred_ans) == normalize(true_ans))

def f1_score(pred_ans, true_ans):
    pred_tokens = normalize(pred_ans).split()
    true_tokens = normalize(true_ans).split()
    comm = sum((Counter(pred_tokens) & Counter(true_tokens)).values())
    if comm == 0:
        f1 = 0
    else:
        p = comm / len(pred_tokens)
        r = comm / len(true_tokens)
        f1 = (2 * p * r) / (p + r)
    return f1

def cal_em_score(pred_ans, true_ans, alias_list=None):
    ans_list = [true_ans]
    if alias_list is not None:
        ans_list += alias_list
    score = max(em_score(pred_ans, alias) for alias in ans_list)
    return score

def cal_f1_score(pred_ans, true_ans, alias_list=None):
    ans_list = [true_ans]
    if alias_list is not None:
        ans_list += alias_list
    score = max(f1_score(pred_ans, alias) for alias in ans_list)
    return score

def batch_eval(true_path, pred_path, out_path, max_size=None):
    true_data = read_jsonl(true_path, max_size=max_size)
    pred_data = read_jsonl(pred_path, max_size=max_size)
    assert len(true_data) == len(pred_data)
    f1_list, em_list = [], []
    eval_data = []
    for true_item, pred_item in zip(true_data, pred_data):
        assert true_item["id"] == pred_item["id"]
        true_ans = true_item["answer"]
        alias = true_item.get("alias", None)
        pred_ans = pred_item["answer"]
        f1 = cal_f1_score(pred_ans, true_ans, alias)
        em = cal_em_score(pred_ans, true_ans, alias)
        eval_item = {"id": true_item["id"], "question": true_item["question"], "true_answer": true_ans, "pred_answer": pred_ans, "f1": f1, "em": em}
        if alias is not None:
            eval_item["alias"] = alias
        eval_data.append(eval_item)
        f1_list.append(f1)
        em_list.append(em)
    save_jsonl(eval_data, out_path)
    f1_score = sum(f1_list) / len(f1_list)
    em_score = sum(em_list) / len(em_list)
    result = {"f1": f1_score, "em": em_score}
    return result

def batch_eval_yn(true_path, pred_path, out_path, max_size=None):
    true_data = read_jsonl(true_path, max_size=max_size)
    pred_data = read_jsonl(pred_path, max_size=max_size)
    assert len(true_data) == len(pred_data)
    acc_list = []
    eval_data = []
    for true_item, pred_item in zip(true_data, pred_data):
        assert true_item["id"] == pred_item["id"]
        true_ans = true_item["answer"]
        raw_pred_ans = normalize(pred_item["answer"])
        if raw_pred_ans in ("yes", "no"):
            pred_ans = raw_pred_ans
        elif raw_pred_ans.startswith("yes"):
            pred_ans = "yes"
        elif raw_pred_ans.startswith("no"):
            pred_ans = "no"
        elif "yes" in raw_pred_ans.split():
            pred_ans = "yes"
        elif "no" in raw_pred_ans.split():
            pred_ans = "no"
        else:
            pred_ans = pred_item["answer"]
        if pred_ans.lower() == true_ans.lower():
            acc = 1
        else:
            acc = 0
        eval_item = {"id": true_item["id"], "question": true_item["question"], "true_answer": true_ans, "pred_answer": pred_ans, "raw_pred_answer": pred_item["answer"], "acc": acc}
        eval_data.append(eval_item)
        acc_list.append(acc)
    save_jsonl(eval_data, out_path)
    acc_score = sum(acc_list) / len(acc_list)
    result = {"acc": acc_score}
    return result
