#!/usr/bin/env bash

#SBATCH --job-name=vLLM
#SBATCH --account=pawsey0001-gpu
#SBATCH --reservation=PAWSEY_GPU_COS_TESTING
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --nodes=$1
#SBATCH --partition=gpu
module load singularity/4.1.0-slurm
srun ~/projects/raycluster/start-cluster.py singularity ~/.opt/vLLM/vllm_latest.sif \
	--slurm --tensor-parallel-size 4 --vllm-port 7777 --pipeline-parallel-size 2 --num-gpus 4 \
	--model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
