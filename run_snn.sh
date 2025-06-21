#!/bin/bash
#SBATCH --job-name=snn_job                # Name of your job
#SBATCH --partition=IMLcuda1             # GPU partition
#SBATCH --nodelist=nodeicuda1            # Specific node
#SBATCH --gres=gpu:RTX4000Ada:1          # Request 1 GPU
#SBATCH --time=01:00:00                  # Max wall time (1 hour)
#SBATCH --output=logs/snn_job_%j.out     # Stdout + stderr log file (%j = job ID)

# Load conda (adjust if needed for your setup)
source ~/.bashrc
conda activate snn

# Optional: print environment info
echo "Running on node: $(hostname)"
nvidia-smi
which python
python --version

# Run your Python script
python ~/sld/main.py
