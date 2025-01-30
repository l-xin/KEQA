# -*- coding: utf-8  -*-

import re
import traceback
from utils import read_jsonl, save_jsonl

def decompose(query, llm, id=None):
    inst = "Your task is to decompose the given question into sub-questions and output these sub-questions sequentially with number. You can use the number of previous sub-questions to replace their answers in the following sub-questions. I will provide you with three questions along with their decomposed sub-questions as examples."
    examples = [
        {
            "question": "Do the anchors on Rede Globo speak Chinese?",
            "sub-questions": ["1. What country broadcasts Rede Globo?", "2. What is the official language of #1?", "3. Is #2 Chinese?"]
        },
        {
            "question": "What nationality was James Henry Miller's wife?",
            "sub-questions": ["1. Who is James Henry Miller's wife?", "2. What nationality was #1?"]
        },
        {
            "question": "Which tennis player was born first, Michael Chang or Sara Errani?",
            "sub-questions": ["1. When was Michael Chang born?", "2. When was Sara Errani born?", "3. Is #1 earlier than #2?"]
        }
    ]
    example_prompt = "\n\n".join(f"Question: {item['question']}\nSub-Questions: " + "\n".join(item["sub-questions"]) for item in examples)
    prompt = f"{inst}\n\n{example_prompt}\n\nQuestions: {query}\nSub-Questions: "
    text = llm(prompt, id=id)
    raw_sub_questions = [line.strip() for line in text.split('\n') if line.strip() != '']
    pattern = re.compile(r"^\d+\s*\.\s*(.*?)$")
    sub_questions = []
    for item in raw_sub_questions:
        m = re.match(pattern, item)
        if m is not None:
            sub_questions.append(m[1])
        else:
            sub_questions.append(item)
    return sub_questions

def batch_decompose(in_path, out_path, llm, max_size=None):
    data = read_jsonl(in_path, max_size=max_size)
    new_data = []
    try:
        for item in data:
            sub_questions = decompose(item["question"], llm, id=f"decomp-{item['id']}")
            new_data.append({"id": item["id"], "question": item["question"], "sub_questions": sub_questions})
    except:
        save_jsonl(new_data, out_path)
        traceback.print_exc()
        exit()
    save_jsonl(new_data, out_path)
    return
