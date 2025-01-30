# -*- coding: utf-8 -*-

import csv
import json
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from parse_args import args
from utils import get_logger

class ESRetriever:
    def __init__(self):
        self.es = Elasticsearch(
            hosts=[args.es_host],
            request_timeout=100,
            retry_on_timeout=True
        )
        self.index_name = args.es_index
        self.corpus_path = args.es_corpus
        self.fields = ["text", "title"]
        self.log = get_logger("ESRetriever", args.es_log)
        
        if not self.es.indices.exists(index=self.index_name).body:
            self.log.info(f"building index from {self.corpus_path} ...")
            self.build_index()
        return
    
    def __del__(self):
        self.es.close()
        return
    
    def __call__(self, query, topk, id=None):
        body = {
            "query": {
                "multi_match": {
                    "query": query, 
                    "type": "best_fields",
                    "fields": self.fields,
                    "tie_breaker": 0.5
                }
            }
        }
        resp = self.es.search(index=self.index_name, search_type="dfs_query_then_fetch", body=body, size=topk)
        results = []
        for hit in resp["hits"]["hits"]:
            result_item = {"id": hit["_id"], "score": hit["_score"]}
            for k in self.fields:
                result_item[k] = hit["_source"][k]
            results.append(result_item)
        self.log.info(json.dumps({"id": id, "topk": topk, "query": query, "results": results}, ensure_ascii=False))
        return results
    
    def build_index(self):
        self.es.indices.delete(index=self.index_name, ignore=[404])
        time.sleep(5)
        mappings = {
            "mappings": {
                "properties": {k: {"type": "text", "analyzer": "english"} for k in self.fields}
            }
        }
        self.es.indices.create(index=self.index_name, body=mappings)
        for idx, (ok, info) in enumerate(streaming_bulk(client=self.es, index=self.index_name, actions=self.generate_actions(self.corpus_path))):
            print(idx)
            if not ok:
                print(info)
        time.sleep(5)
        return
    
    def generate_actions(self, corpus_path):
        with open(corpus_path, "rt", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter='\t')
            header = next(reader)
            for row in reader:
                es_doc = dict(zip(header, row))
                es_doc["_id"] = es_doc.pop("id")
                es_doc["_op_type"] = "index"
                es_doc["refresh"] = "wait_for"
                yield es_doc
        return

es = ESRetriever()
