# -*- coding: utf-8 -*-

from collections import Counter

from utils import normalize, get_first_line

def compare(question, answer1, answer2, llm, id=None):
    if normalize(answer1) == normalize(answer2):
        s_result = "Yes"
    else:
        inst = "Your task is to compare whether the Answer 1 and Answer 2 have similar meanings to the Question. You should report the comparison result with \"Yes\" or \"No\"."
        prompt = f"{inst}\n\nQuestion: {question}\n\nAnswer 1: {answer1}\n\nAnswer 2: {answer2}\n\nComparison Result: "
        s_result = llm(prompt, id=id)
        s_result = get_first_line(s_result)
    if s_result.lower().startswith("yes"):
        result = True
    elif s_result.lower().startswith("no"):
        result = False
    elif "yes" in s_result.lower():
        result = True
    else:
        result = False
    return result, s_result

def group(question, answer_list, llm, sc_threshold, id=None):
    compare_result = []
    answer_group = dict()
    agree_answer = None
    for ans_idx, answer in enumerate(answer_list):
        if answer.lower() == "unknown":
            continue
        hit = None
        for anchor, ans_list in answer_group.items():
            if answer in ans_list:
                hit = anchor
                break
        if hit is None:
            for anc_idx, anchor in enumerate(answer_group.keys()):
                cmp, s_cmp = compare(question, answer, anchor, llm, id=f"cmp-{ans_idx+1}-{anc_idx+1}-{id}")
                compare_result.append({"answer1": answer, "answer2": anchor, "label": cmp, "s_label": s_cmp})
                if cmp:
                    hit = anchor
                    break
        if hit is None:
            hit = answer
        answer_group.setdefault(hit, []).append(answer)
        new_anchor = Counter(answer_group[hit]).most_common(1)[0][0]
        if new_anchor != hit:
            answer_group[new_anchor] = answer_group.pop(hit)
            hit = new_anchor
        if len(answer_group[hit]) >= sc_threshold:
            agree_answer = hit
            break
        elif max(len(g) for g in answer_group.values()) + (len(answer_list) - ans_idx - 1) < sc_threshold:
            break
    answer_group = sorted(answer_group.values(), key=lambda g: len(g), reverse=True)
    return agree_answer, answer_group, compare_result
