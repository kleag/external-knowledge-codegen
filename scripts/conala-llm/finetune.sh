#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

logs_dir="logs/conala-llm"
model_dir="saved_models/conala-llm"
decodes_dir="decodes/conala-llm"
data_dir="data/conala-llm"
scripts_dir="scripts/conala-llm"
install -d ${decodes_dir}
install -d ${logs_dir}
install -d ${model_dir}
install -d ${data_dir}

seed=0
mined_num=$1
pretrained_model_name=$2
freq=${3:-3}
vocab="${data_dir}/vocab.src_freq${freq}.code_freq${freq}.mined_${mined_num}.bin"
finetune_file="${data_dir}/train.var_str_sep.bin"
dev_file="${data_dir}/dev.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=finetune.conala.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.seed${seed}.pre_${mined_num}

echo "**** Writing results to ${logs_dir}/${model_name}.log ****"
echo commit hash: "$(git rev-parse HEAD)" > ${logs_dir}/"${model_name}".log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --evaluator conala_evaluator \
    --asdl_file src/asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${finetune_file} \
    --dev_file ${dev_file} \
    --pretrain ${pretrained_model_name} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --max_epoch 80 \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to ${model_dir}/${model_name} 2>&1 | tee ${logs_dir}/${model_name}.log

bash ${scripts_dir}/test.sh ${model_dir}/${model_name}.bin 2>&1 | tee -a ${logs_dir}/${model_name}.log
