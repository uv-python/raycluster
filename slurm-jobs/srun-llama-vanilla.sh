#!/usr/bin/env bash
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
srun -u ~/projects/raycluster/start-cluster.py singularity ~/.opt/vLLM/vllm_latest.sif \
	--container-args "--bind ./app:/app" \
	--slurm --auto --model 'meta-llama/Llama-4-Maverick-17B-128E' --max-model-len 200000

# 68 tokens/second
