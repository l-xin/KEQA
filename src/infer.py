# -*- coding: utf-8 -*-

import traceback

from group import group
from qa import quiz, rag, rag_summarize
from robust_retrieve import robust_retrieve
from utils import read_jsonl, save_jsonl

def infer(query, decomp_list, llm, doc_retriever, slm, refer_retriever, sc_num, sc_threshold, n_doc, n_refer, min_refer_label_num, yn=False, id=None):
    subqa_list = []
    subqa_results = []
    for subq_idx, subq in enumerate(decomp_list):
        for ans_idx in range(subq_idx):
            placeholder = f"#{ans_idx+1}"
            if placeholder in subq:
                subq = subq.replace(placeholder, f"{subqa_list[ans_idx]['answer']}")
        sc_answer_list = []
        sc_result = []
        for sc_idx in range(sc_num):
            sc_answer, sc_raw_answer = quiz(subq, llm, id=f"inf-sc-{subq_idx+1}-{sc_idx+1}-{id}")
            sc_answer_list.append(sc_answer)
            sc_result_item = {"answer": sc_answer}
            if sc_raw_answer != sc_answer:
                sc_result_item["raw_answer"] = sc_raw_answer
            sc_result.append(sc_result_item)
        agree_answer, answer_group, group_result = group(subq, sc_answer_list, slm, sc_threshold, id=f"inf-{subq_idx+1}-{id}")
        if agree_answer is not None and agree_answer != '' and agree_answer.lower() != "unknown":
            sub_answer = agree_answer
            sub_raw_answer = None
            util_result = None
        else:
            docs, util_result = robust_retrieve(subq, slm, doc_retriever, refer_retriever, n_doc, n_refer, min_refer_label_num, id=f"inf-{subq_idx+1}-{id}")
            if len(docs) > 0:
                sub_answer, sub_raw_answer = rag(subq, docs, llm, id=f"inf-rag-{subq_idx+1}-{id}")
            else:
                if len(answer_group) > 0:
                    sub_answer = answer_group[0][0]
                    sub_raw_answer = None
                else:
                    sub_answer = "unknown"
                    sub_raw_answer = None
        subqa_list.append({"subq": subq, "answer": sub_answer})
        subqa_result_item = {"subq": subq, "answer": sub_answer}
        if sub_raw_answer is not None:
            subqa_result_item["raw_answer"] = sub_raw_answer
        subqa_result_item["answer_group"] = answer_group
        subqa_result_item["sc"] = sc_result
        subqa_result_item["group"] = group_result
        if util_result is not None:
            subqa_result_item["util"] = util_result
        subqa_results.append(subqa_result_item)
    
    if len(decomp_list) == 1:
        answer = subqa_list[0]["answer"]
        raw_answer = None
    else:
        answer, raw_answer = rag_summarize(query, subqa_list, llm, yn=yn, id=f"inf-con-{id}")
    infer_result = {"answer": answer}
    if raw_answer is not None:
        infer_result["raw_answer"] = raw_answer
    infer_result["subqa"] = subqa_results
    return answer, infer_result

def batch_infer(decomp_path, out_path, llm, doc_retriever, slm, refer_retriever, sc_num, sc_threshold, n_doc, n_refer, min_refer_label_num, yn=False, max_size=None):
    data = read_jsonl(decomp_path, max_size=max_size)
    new_data = []
    try:
        for item in data:
            if "sub_questions" in item.keys():
                decomp_list = item["sub_questions"]
            else:
                decomp_list = [item["question"]]
            answer, infer_result = infer(item["question"], decomp_list, llm, doc_retriever, slm, refer_retriever, sc_num, sc_threshold, n_doc, n_refer, min_refer_label_num, yn=yn, id=item["id"])
            new_item = {"id": item["id"], "question": item["question"], "answer": answer, "infer": infer_result}
            new_data.append(new_item)
    except:
        save_jsonl(new_data, out_path)
        traceback.print_exc()
        exit()
    save_jsonl(new_data, out_path)
    return
