# -*- coding: utf-8 -*-

import json
import torch
import transformers

from parse_args import args
from utils import get_logger

class LlamaClient:
    def __init__(self):
        self.device = "cuda" if args.cuda is not None and torch.cuda.is_available() else "cpu"
        self.pipeline = transformers.pipeline("text-generation", model=args.llama_model, torch_dtype=torch.bfloat16, device=self.device)
        self.terminators = [self.pipeline.tokenizer.eos_token_id, self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.log = get_logger("LlamaClient", args.llama_log)
        return

    def __call__(self, query, id=None):
        # messages = [{"role": "user", "content": query}]
        # prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = query
        res = self.pipeline(prompt, eos_token_id=self.terminators)
        text = res[0]["generated_text"][len(prompt):].strip()
        self.log.info(json.dumps({"id": id, "query": query, "text": text}))
        return text

llama = LlamaClient()
