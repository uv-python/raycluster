#!/usr/bin/env bash
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
srun -u ~/projects/raycluster/start-cluster.py singularity ~/.opt/vLLM/vllm_latest.sif \
	--container-args "--bind ./app:/app" \
	--slurm --auto --model deepseek-ai/DeepSeek-R1-0528 --max-model-len 150000 \
	--enable-expert-parallel --gpu-memory-utilization 0.98 --enforce-eager --dtype bfloat16
