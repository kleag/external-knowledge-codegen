#!/bin/bash

#SBATCH --job-name=conalapt
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH --nodes=1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

# #SBATCH --partition=amd
# #SBATCH --partition=classicgpu
# #SBATCH --partition=gpu40G
#SBATCH --partition=gpu80G
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpup5000short
# #SBATCH --partition=gpuv100
# #SBATCH --partition=lasti
# #SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
# #SBATCH --reservation=root_52

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=120G


#set -o nounset
set -o errexit
set -o pipefail

echo "$0"

# activate environments
source /home/users/gdechalendar/miniconda3/bin/activate
source activate tranx

/usr/bin/env python3 --version
if [[ -v SLURM_JOB_ID ]] ; then
  nvidia-smi

  # Affiche la (ou les gpus) allouee par Slurm pour ce job
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

echo "Begin on machine: `hostname`"

# conda list

EXECUTOR=srun
if [ -z ${SLURM_JOB_ID+x} ]; then
  echo "Not in sbatch"
  EXECUTOR=
fi
echo "EXECUTOR is ${EXECUTOR}"


##############

#echo 'Syncing data'
#install -d /scratch/gael/concode
#rsync -avz --inplace --delete-delay bergamote2-ib:/scratch_global/gael/concode/data/d_100k_762 /scratch/gael/concode/
#ls /scratch/gael/concode

echo 'Script starting'

cd /home/users/gdechalendar/Projets/DeepLab/external-knowledge-codegen
export PYTHONPATH=$PWD:$PWD/src
# run script
echo 'Pretraining'
${EXECUTOR} bash scripts/conala/train_pre.sh 100000 3

#echo 'Fine tuning'
#${EXECUTOR} scripts/conala/finetune_retrieved_distsmpl.sh 100000 snippet_count100k_topk1_temp2 saved_models/conala/retdistsmpl.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.vocab.src_freq3.code_freq3.mined_100000.goldmine_snippet_count100k_topk1_temp2.bin.pre_100000_goldmine_snippet_count100k_topk1_temp2.bin.seed0.bin

wait

#echo 'Syncing learnt models'
#rsync -avz --inplace /scratch/gael/concode/d_100k_762/concode bergamote2-ib:/scratch_global/gael/concode/data/d_100k_762
#ls /scratch/gael/concode

echo "Slurm script Done."

