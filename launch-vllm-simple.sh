#!/usr/bin/env bash

#SBATCH --job-name=vLLM
#SBATCH --account=pawsey0001-gpu
#SBATCH --reservation=PAWSEY_GPU_COS_TESTING
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2
#SBATCH --partition=gpu
module load singularity/4.1.0-slurm
srun ~/projects/raycluster/start-cluster.py singularity ~/.opt/vLLM/vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif \
	--slurm --auto --model Qwen/Qwen3-30B-A3B
