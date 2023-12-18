#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

logs_dir="logs/conala"
model_dir="saved_models/conala"
decodes_dir="decodes/conala"
data_dir="data/conala"

cuda=""
# cuda="--cuda"
seed=0
vocab="${data_dir}/vocab.src_freq3.code_freq3.bin"
train_file="${data_dir}/train.all_0.bin"
dev_file="${data_dir}/dev.bin"

batch_size=256
hidden_size=768
embed_size=768
action_embed_size=768
field_embed_size=768
type_embed_size=768
# field_embed_size 256 --type_embed_size 256

dropout=0.3
lr=0.001
lr_decay=0.5
max_epoch=80
beam_size=15
lstm='lstm'  # lstm
encoder='bert' # lstm or bert
lr_decay_after_epoch=15
model_name=conala.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).glorot.par_state.seed${seed}

install -d ${logs_dir}
install -d ${model_dir}
install -d ${decodes_dir}

echo "**** Writing results to ${logs_dir}/${model_name}.log ****"
echo commit hash: `git rev-parse HEAD` > ${logs_dir}/${model_name}.log

#     -m pdb \
python -u  \
    exp.py \
    ${cuda} \
    --data_path ${data_dir} \
    --seed ${seed} \
    --mode train \
    --evaluator conala_evaluator \
    --asdl_file src/asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --encoder ${encoder} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --batch_size ${batch_size} \
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
    --max_epoch ${max_epoch} \
    --beam_size ${beam_size} \
    --log_every 1 \
    --save_decode_to ${decodes_dir}/${model_name}.decode \
    --save-to ${model_dir}/${model_name} \
    2>&1 | tee ${logs_dir}/${model_name}.log
#     --verbose \

echo
echo "Trainin done. Now testing."
. scripts/conala/test.sh ${model_dir}/${model_name}.bin 2>&1 | tee -a ${logs_dir}/${model_name}.log
