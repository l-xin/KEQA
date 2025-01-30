# -*- coding: utf-8 -*-

import csv
import json
import os
import random

def read_json(path):
    with open(path, "rt", encoding="utf-8") as file:
        obj = json.load(file)
    return obj

def read_jsonl(path):
    with open(path, "rt", encoding="utf-8") as file:
        obj_list = [json.loads(line) for line in file]
    return obj_list

def save_jsonl(obj_list, path):
    with open(path, "wt", encoding="utf-8") as file:
        for obj in obj_list:
            file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    return

def convert_nq_qa(in_root, out_root):
    io_map = {
        "train.query.txt": ("train.full.jsonl", "train.answers.txt"),
        "test.query.txt": ("dev.full.jsonl", "test.answers.txt")
    }
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    for in_query_filename, (out_filename, in_ans_filename) in io_map.items():
        in_query_path = os.path.join(in_root, in_query_filename)
        in_ans_path = os.path.join(in_root, in_ans_filename)
        out_path = os.path.join(out_root, out_filename)
        with open(in_query_path, "rt", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter='\t')
            query_data = [row for row in reader]
        with open(in_ans_path, "rt", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter='\t')
            ans_data = [row for row in reader]
        assert len(query_data) == len(ans_data)
        data = []
        for query_row, ans_row in zip(query_data, ans_data):
            query_id, query = query_row
            ans_id = ans_row[0]
            ans = ans_row[1]
            alias = ans_row[2:]
            assert query_id == ans_id
            data.append({"id": str(query_id), "question": query, "answer": ans, "alias": alias})
        save_jsonl(data, out_path)
    return

def convert_strategy_qa(in_root, out_root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    in_path = os.path.join(in_root, "strategyqa_train.json")
    out_path = os.path.join(out_root, "train.full.jsonl")
    raw_data = read_json(in_path)
    out_data = []
    for item in raw_data:
        id = item["qid"]
        question = item["question"]
        answer = "Yes" if item["answer"] else "No"
        new_item = {"id": id, "question": question, "answer": answer}
        out_data.append(new_item)
    save_jsonl(out_data, out_path)
    return

def convert_hotpotqa_qa(in_root, out_root):
    io_map = {
        "hotpot_train_v1.1.json": "train.full.jsonl",
        "hotpot_dev_fullwiki_v1.json": "dev.full.jsonl"
    }
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    for in_filename, out_filename in io_map.items():
        in_path = os.path.join(in_root, in_filename)
        out_path = os.path.join(out_root, out_filename)
        data = read_json(in_path)
        data = [{"id": item["_id"], "question": item["question"], "answer": item["answer"]} for item in data]
        save_jsonl(data, out_path)
    return

def convert_2wikimultihopqa_qa(in_root, out_root):
    alias_path = os.path.join(in_root, "id_aliases.json")
    alias_dict = {item["Q_id"]: item for item in read_jsonl(alias_path)}
    io_map = {
        "train.json": "train.full.jsonl",
        "dev.json": "dev.full.jsonl"
    }
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    for in_filename, out_filename in io_map.items():
        in_path = os.path.join(in_root, in_filename)
        out_path = os.path.join(out_root, out_filename)
        data = read_json(in_path)
        new_data = []
        for item in data:
            new_item = {"id": item["_id"], "question": item["question"], "answer": item["answer"]}
            if item["answer_id"] is None:
                new_item["alias"] = []
            else:
                alias_data = alias_dict[item["answer_id"]]
                new_item["alias"] = alias_data["aliases"] + alias_data["demonyms"]
            new_data.append(new_item)
        save_jsonl(new_data, out_path)
    return

def sample_test_set(in_path, out_path, n_sample=500):
    all_data = read_jsonl(in_path)
    out_data = random.sample(all_data, n_sample)
    question_set = set(item["question"] for item in out_data)
    assert len(question_set) == len(out_data)
    save_jsonl(out_data, out_path)
    return

def fetch_rest_set(all_path, test_path, out_path):
    all_data = read_jsonl(all_path)
    test_data = read_jsonl(test_path)
    test_ids = set(item["id"] for item in test_data)
    rest_data = [item for item in all_data if item["id"] not in test_ids]
    assert len(test_data) + len(rest_data) == len(all_data)
    save_jsonl(rest_data, out_path)
    return

if __name__ == "__main__":
    convert_nq_qa("data/raw/NQ", "data/NQ")
    sample_test_set("data/NQ/dev.full.jsonl", "data/NQ/test.jsonl", n_sample=500)
    sample_test_set("data/NQ/train.full.jsonl", "data/NQ/ref.jsonl", n_sample=500)
    
    convert_strategy_qa("data/raw/StrategyQA", "data/StrategyQA")
    sample_test_set("data/StrategyQA/train.full.jsonl", "data/StrategyQA/test.jsonl", n_sample=500)
    fetch_rest_set("data/StrategyQA/train.full.jsonl", "data/StrategyQA/test.jsonl", "data/StrategyQA/train.rest.jsonl")
    sample_test_set("data/StrategyQA/train.rest.jsonl", "data/StrategyQA/ref.jsonl", n_sample=500)
    
    convert_hotpotqa_qa("data/raw/HotpotQA", "data/HotpotQA")
    sample_test_set("data/HotpotQA/dev.full.jsonl", "data/HotpotQA/test.jsonl", n_sample=500)
    sample_test_set("data/HotpotQA/train.full.jsonl", "data/HotpotQA/ref.jsonl", n_sample=500)
    
    convert_2wikimultihopqa_qa("data/raw/2WikiMultihopQA", "data/2WikiMultihopQA")
    sample_test_set("data/2WikiMultihopQA/dev.full.jsonl", "data/2WikiMultihopQA/test.jsonl", n_sample=500)
    sample_test_set("data/2WikiMultihopQA/train.full.jsonl", "data/2WikiMultihopQA/ref.jsonl", n_sample=500)
