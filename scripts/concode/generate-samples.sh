#! /bin/bash

set -o errexit
set -o pipefail
set -o nounset

set -x

python apidocs/retrieve.py --index_name "java-code" --method topk --inp data/concode/concode_train.json --out output-topk-snippet.jsonl  --field snippet

python apidocs/retrieve.py --index_name "java-code" --method topk --inp data/concode/concode_train.json --out output-topk-intent.jsonl  --field intent

python apidocs/retrieve.py --index_name "java-code" --method dist --inp data/concode/concode_train.json --out dist-snippet.csv  --field snippet --max_count 39674 --temp 2

python apidocs/retrieve.py --index_name "java-code" --method dist --inp data/concode/concode_train.json --out dist-intent.csv  --field intent --max_count 39674 --temp 2

python apidocs/retrieve.py --index_name "java-code" --method sample --inp dist-snippet.csv:data/concode/concode_train.json --out output-sample-snippet.jsonl  --field snippet --max_count 39674 --temp 2

python apidocs/retrieve.py --index_name "java-code" --method sample --inp dist-intent.csv:data/concode/concode_train.json --out output-sample-intent.jsonl  --field snippet --max_count 39674 --temp 2
