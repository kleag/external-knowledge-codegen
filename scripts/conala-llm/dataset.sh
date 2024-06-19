#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

python src/datasets/conala/dataset.py --mined data/conala-llm/conala-mined.jsonl --num_mined 100000 --include_api apidocs/processed/distsmpl/snippet_15k/goldmine_snippet_count100k_topk1_temp2.jsonl --tokenizer nltk --no_rewritten --out-dir data/conala-llm --train data/conala-llm/conala-train.json --test data/conala-llm/conala-test.json --intent snippet
