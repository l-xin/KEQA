 # Explore What LLM Does Not Know in Complex Question Answering
Source code for paper *Explore What LLM Does Not Know in Complex Question Answering*.

 ## Dependencies
- python

- elasticsearch
- faiss-cpu
- numpy
- openai
- torch
- transformers

 ## Usage
- Dataset preparation: Download [NaturalQuestions](https://rocketqa.bj.bcebos.com/corpus/nq.tar.gz), [StrategyQA](https://allenai.org/data/strategyqa), [HotpotQA](https://hotpotqa.github.io), [2WikiMultihopQA](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip), place the decompressed files into `data/raw/NQ`, `data/raw/StrategyQA`, `data/raw/HotpotQA`, `data/raw/2WikiMultihopQA` respectively, and run `src/dataproc.py` to preprocess the datasets.
```bash
python3 src/dataproc.py
```
- Corpus preparation: Download [Wikipedia dump](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz) from Dec 20, 2018, and decompress into `data/corpus/psgs_w100.tsv`.
- Retriever preparation: Download [Elasticsearch](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.13.0-linux-x86_64.tar.gz), decompress the file, and start ES service.
```bash
nohup ./elasticsearch/bin/elasticsearch &> es_server.log &
```
Note that `elasticsearch` python interface should match the version of ES (8.13.0 in our setting). In case you can not start ES service, you can cancel safety authentication by disabling `xpack.security.enabled` and `xpack.security.http.ssl` in `elasticsearch/config/elasticsearch.yml` for *experimental usage*. 

- Run `src/run_refer.py` to build utility reference.
```bash
python3 src/run_refer.py --gpt-key <your-openai-key>
```
- Run `src/run_infer.py` to run the QA experiments.
```bash
python3 src/run_infer.py --gpt-key <your-openai-key> --cuda 0
``` 
It may take much time to build index for ES and FAISS in first run. For more running arguments, please refer to [src/parse_args.py](src/parse_args.py).