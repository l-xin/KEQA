# -*- coding: utf-8 -*-

from utils import get_first_line, extract_answer

def quiz(query, llm, yn=False, id=None):
    if yn:
        inst = "Your task is to answer the question with \"Yes\" or \"No\"."
    else:
        inst = "Your task is to answer the question with a short phrase."
    if not query.endswith("?"):
        query += "?"
    prompt = f"{inst}\n\nQuestion: {query}\nAnswer: "
    raw_answer = llm(prompt, id=id)
    answer = get_first_line(raw_answer)
    return answer, raw_answer

def quiz_cot(query, llm, yn=False, id=None):
    if yn:
        inst = "Your task is to answer the question by thinking step-by-step, and point out the answer as \"Yes\" or \"No\" with \"the answer is\"."
    else:
        inst = "Your task is to answer the question by thinking step-by-step, and point out one short phrase as the answer with \"the answer is\"."
    if not query.endswith("?"):
        query += "?"
    prompt = f"{inst}\n\nQuestion: {query}\nAnswer: "
    raw_answer = llm(prompt, id=id)
    answer = extract_answer(raw_answer)
    return answer, raw_answer

def rag(query, docs, llm, yn=False, id=None):
    if yn:
        inst = "Your task is to refer to the passages and answer the question with \"Yes\" or \"No\"."
    else:
        inst = "Your task is to refer to the passages and answer the question with a short phrase."
    if not query.endswith("?"):
        query += "?"
    if len(docs) == 1:
        doc_prompt = f"Passage: {docs[0]}"
    else:
        doc_prompt = "\n\n".join(f"Passage {doc_idx+1}: {doc}" for doc_idx, doc in enumerate(docs))
    prompt = f"{inst}\n\n{doc_prompt}\n\nQuestion: {query}\n\nAnswer: "
    raw_answer = llm(prompt, id=id)
    answer = get_first_line(raw_answer)
    return answer, raw_answer

def rag_cot(query, docs, llm, yn=False, id=None):
    if yn:
        inst = "Your task is to refer to the passages and answer the question by thinking step-by-step, and point out the answer as \"Yes\" or \"No\" with \"the answer is\"."
    else:
        inst = "Your task is to refer to the passages and answer the question by thinking step-by-step, and point out one short phrase as the answer with \"the answer is\"."
    if not query.endswith("?"):
        query += "?"
    if len(docs) == 1:
        doc_prompt = f"Passage: {docs[0]}"
    else:
        doc_prompt = "\n\n".join(f"Passage {doc_idx+1}: {doc}" for doc_idx, doc in enumerate(docs))
    prompt = f"{inst}\n\n{doc_prompt}\n\nQuestion: {query}\n\nAnswer: "
    raw_answer = llm(prompt, id=id)
    answer = extract_answer(raw_answer)
    return answer, raw_answer

def rag_summarize(query, subqa_list, llm, yn=False, id=None):
    if yn:
        inst = "Your task is to summarize these information to answer the question by thinking step-by-step, and point out the answer as \"Yes\" or \"No\" with \"the answer is\"."
        examples = [
            {
                "question": "Are both Kurram Garhi and Trojkrsti located in the same country?",
                "answer": "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no."
            },
            {
                "question": "Are Hyomin and Bobby Gillespie both members of a band?",
                "answer": "Hyomin is a member of K-pop group, T-ara. Bobby Gillespie is a member of rock band Primal Scream. Thus, they are both members of a band. So the answer is yes."
            }
        ]
    else:
        inst = "Your task is to summarize these information to answer the question by thinking step-by-step, and point out one short phrase as the answer with \"the answer is\"."
        examples = [
            {
                "question": "When did the director of film Hypocrite (Film) die?",
                "answer": "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013."
            },
            {
                "question": "Are both Kurram Garhi and Trojkrsti located in the same country?",
                "answer": "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no."
            }
        ]
    example_prompt = "\n\n".join(f"Question: {example['question']}\nAnswer: {example['answer']}" for example in examples)
    if not query.endswith("?"):
        query += "?"
    subqa_prompt = "\n\n".join(f"Sub-Question {idx+1}: {subqa['subq']}\nAnswer {idx+1}: {subqa['answer']}" for idx, subqa in enumerate(subqa_list))
    prompt = f"Given the sub-questions and corresponding answers of one question as follows:\n\n{subqa_prompt}\n\n{inst}\nFor example,\n\n{example_prompt}\n\nQuestion: {query}\nAnswer: "
    raw_answer = llm(prompt, id=id)
    answer = extract_answer(raw_answer)
    return answer, raw_answer
