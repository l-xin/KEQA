# -*- coding: utf-8 -*-

from utils import get_first_line

def check_utility(query, doc, refers, llm, id=None):
    inst = "Given one passage and one question, your task is to judge whether the passage contains information to answer the question. You should report the judgment result with \"Yes\" or \"No\". I will provide you with some passages along with questions and judgment results as examples."
    label_map = {1: "Yes", -1: "No"}
    refer_prompt = "\n\n".join(f"Passage: {refer['text']}\nQuestion: {refer['question']}\nJudgment Result: {label_map[refer['label']]}" for refer in refers)
    prompt = f"{inst}\n\n{refer_prompt}\n\nPlease judge whether the passage contains information to answer the question below following previous examples.\n\nPassage: {doc}\nQuestion: {query}\nJudgment Result: "
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

def robust_retrieve(query, llm, doc_retriever, refer_retriever, n_doc, n_refer, min_refer_label_num, id=None):
    cand_docs = doc_retriever(query=query, topk=n_doc, id=f"util-{id}")
    cand_docs = [doc["text"] for doc in cand_docs]
    if len(cand_docs) > 0:
        all_refer_data = refer_retriever(query, cand_docs, n_refer, min_refer_label_num, id=f"util-{id}")
        assert len(all_refer_data) == len(cand_docs)
        util_docs = []
        util_results = []
        for doc_idx, (doc, refer_data) in enumerate(zip(cand_docs, all_refer_data)):
            result, s_result = check_utility(query, doc, refer_data["refer"], llm, id=f"util-{doc_idx+1}-{id}")
            if result:
                util_docs.append(doc)
            util_results.append({"doc": doc, "label": result, "s_label": s_result, "refer": refer_data["refer"]})
    else:
        util_docs = []
        util_results = []
    return util_docs, util_results
