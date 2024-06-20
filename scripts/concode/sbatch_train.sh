#!/bin/bash
#SBATCH --job-name=run         # Job name
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --partition=cpu           # Specify the CPU partition
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --mem=100GB
#SBATCH  --cpus-per-task=64

bash scripts/concode/train.sh