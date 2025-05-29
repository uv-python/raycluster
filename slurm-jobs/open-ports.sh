#!/usr/bin/env bash

#SBATCH --job-name="All to all multicast test"
#SBATCH --account=pawsey0001-gpu
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --nodes=16
#SBATCH --partition=gpu
srun netstat -na | grep "6379"
#srun $HOME/projects/raycluster/slurm-all-to-all-udp-test.py
