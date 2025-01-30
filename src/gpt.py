# -*- coding: utf-8 -*-

import json
import time
import openai

from parse_args import args
from utils import get_logger

class ChatGPTClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=args.gpt_key, base_url=args.gpt_url)
        self.model = args.gpt_model
        self.n_try = 5
        self.log = get_logger("ChatGPTClient", args.gpt_log)
        return
    
    def __del__(self):
        self.client.close()
        return
    
    def __call__(self, query, id=None):
        messages = [{"role": "user", "content": query}]
        text = None
        for i in range(self.n_try):
            if i > 0:
                time.sleep(2)
                self.log.info(f"retry {id} ...")
            try:
                res = self.client.chat.completions.create(messages=messages, model=self.model)
                text = res.choices[0].message.content
                self.log.info(json.dumps({"id": id, "query": query, "text": text, "res": res.model_dump()}, ensure_ascii=False))
                break
            except:
                pass
        if text is None:
            text = ""
        return text

gpt = ChatGPTClient()
