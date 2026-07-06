#!/bin/bash
#SBATCH --job-name=test_singularity
#SBATCH --output=/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness/logs/slurm/test_singularity.%j.out
#SBATCH --error=/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness/logs/slurm/test_singularity.%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1

singularity exec /scratch/mki4895/containers/evalplus-evaluate.sif python -m evalplus.evaluate --help
