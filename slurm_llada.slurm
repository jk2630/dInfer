#!/bin/bash


#SBATCH --job-name=llada_profile
#SBATCH --output=/lustre/nvwulf/scratch/jdharmana/dInfer/logs/llada_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/jdharmana/dInfer/logs/llada_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -p h200x4
#SBATCH --mem=25g
#SBATCH --gpus=1

module load cuda12.8/toolkit/12.8.1 
source /lustre/nvwulf/scratch/jdharmana/dInfer/dinfer_venv/bin/activate

mkdir -p /lustre/nvwulf/scratch/jdharmana/dInfer/profiles
mkdir -p /lustre/nvwulf/scratch/jdharmana/dInfer/logs

nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-devices=0 \
    --force-overwrite=true \
    -o /lustre/nvwulf/scratch/jdharmana/dInfer/profiles/llama_text \
    python /lustre/nvwulf/scratch/jdharmana/dInfer/server/test_llada.py