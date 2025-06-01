#!/usr/bin/env bash

#SBATCH --job-name=vLLM
#SBATCH --account=pawsey0001-gpu
#SBATCH --exclusive
#SBATCH --reservation=PAWSEY_GPU_COS_TESTING
#SBATCH --time=1-00:00:00
#SBATCH --nodes=4
#SBATCH --partition=gpu
module load singularity/4.1.0-slurm
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
srun ~/projects/raycluster/start-cluster.py singularity ~/.opt/vLLM/vllm_latest.sif \
	--slurm --auto --model meta-llama/Llama-4-Maverick-17B-128E --max-model-length 430000
