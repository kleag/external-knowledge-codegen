#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

logs_dir="logs/conala-llm"
model_dir="saved_models/conala-llm"
decodes_dir="decodes/conala-llm"
data_dir="data/conala-llm"
install -d ${decodes_dir}
install -d ${logs_dir}
install -d ${model_dir}
install -d ${data_dir}

cuda=""
# cuda="--cuda"

test_file="${data_dir}/test.bin"
reranker_file="best_pretrained_models/reranker.conala.vocab.src_freq3.code_freq3.mined_100000.intent_count100k_topk1_temp5.bin"
python exp.py \
    ${cuda} \
    --mode test \
    --load_model $1 \
    --load_reranker $reranker_file \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --save_decode_to ${decodes_dir}/$(basename $1).test.decode \
    --decode_max_time_step 100

