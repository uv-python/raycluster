#!/usr/bin/env bash

#SBATCH --job-name=vLLM
#SBATCH --account=pawsey0001-gpu
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --nodes=16
#SBATCH --partition=gpu
module load singularity/4.1.0-slurm
srun ~/projects/raycluster/start-cluster.py singularity ~/.opt/vLLM/vllm_latest.sif \
	--slurm --auto --model Qwen/Qwen3-235B-A22B --gpu-memory-utilization 0.98 --enable-expert-parallel
