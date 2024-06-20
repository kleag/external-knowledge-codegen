#!/bin/bash
#SBATCH --job-name=run         # Job name
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --partition=gpu80G           # Specify the GPU partition
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --mem=150GB
#SBATCH --gres=gpu:1             # number of gpus per node

nvcc -V
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"



python src/datasets/conala/llm_preprocess_generate.py



# python src/datasets/concode/dataset.py 