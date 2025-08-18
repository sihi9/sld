#!/bin/bash
#SBATCH --job-name=snn                   # Name of your job
#SBATCH --partition=IMLcuda3            # GPU partition
#SBATCH --nodelist=nodeicuda3            # Specific node
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --time=23:59:00                  # Max wall time
#SBATCH --output=logs/snn_job_%j.out     # Stdout + stderr log file (%j = job ID)

# Load conda (adjust if needed for your setup)
source ~/.bashrc
conda activate snn

# Optional: print environment info
echo "Running on node: $(hostname)"

# Run your Python script
python ~/sld/main.py "$@"
