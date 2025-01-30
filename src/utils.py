# -*- coding: utf-8 -*-

import json
import logging
import re
import string

def get_logger(name, path):
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

def read_jsonl(path, max_size=None):
    with open(path, "rt", encoding="utf-8") as file:
        obj_list = [json.loads(line) for line in file]
    if max_size is not None:
        obj_list = obj_list[:max_size]
    return obj_list

def save_jsonl(obj_list, path):
    with open(path, "wt", encoding="utf-8") as file:
        for obj in obj_list:
            file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    return

def get_first_line(s):
    line_list = [l.strip() for l in s.split('\n') if l.strip() != '']
    if len(line_list) > 0:
        line = line_list[0]
    else:
        line = ''
    return line

def extract_answer(answer):
    final_answer = None
    ans_lines = [ans_line.strip() for ans_line in answer.split('\n') if ans_line.strip() != '']
    for ans_line in ans_lines:
        match = re.match(r"^.*?answer is\s?:? (.*?)\.?$", ans_line, re.S)
        if match is not None:
            final_answer = match.group(1).strip()
            break
    if final_answer is None or final_answer == '':
        if len(ans_lines) > 0:
            final_answer_list = [l.strip() for l in ans_lines[-1].split('. ') if l.strip() != '']
            if len(final_answer_list) > 0:
                final_answer = final_answer_list[-1]
            else:
                final_answer = ''
        else:
            final_answer = ''
    return final_answer

def normalize(text):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", ' ', s)

    def white_space_fix(s):
        return ' '.join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else ' ' for ch in s)

    def lower(s):
        return s.lower()
    return white_space_fix(remove_articles(remove_punc(lower(text))))
