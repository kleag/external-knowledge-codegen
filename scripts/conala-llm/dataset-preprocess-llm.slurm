#!/bin/bash

#SBATCH --job-name=conalallm
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH --nodes=1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

# #SBATCH --partition=amd
# #SBATCH --partition=classicgpu
# #SBATCH --partition=gpu40G
# #SBATCH --partition=gpu80G
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpup5000short
# #SBATCH --partition=gpuv100
# #SBATCH --partition=lasti
#SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
# #SBATCH --reservation=root_52

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

#SBATCH --time=0-03:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=120G

set -o errexit
set -o pipefail
set -o nounset

source ~/.virtualenvs/ekcg/bin/activate

python src/datasets/conala/llm_preprocess_generate.py
