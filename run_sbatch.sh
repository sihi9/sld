#!/bin/bash

# sbatch run_snn.sh \
#     --model unet \
#     --data carla \
#     --description "Basic analog with new carla every 10th rerun 2" \
#     --features 32 64 128 256 \
#     --initial-scaling 4 \
#     --downscale 1 \
#     --analog \
#     --soft-reset \
#     --no-fc-bottleneck \
#     --no-fc-recurrent \
#     --no-conv-recurrent \


sbatch run_snn.sh \
    --model unet \
    --data carla \
    --description "best with new carla every 10th rerun 2" \
    --features 16 24 32 48 \
    --initial-scaling 4 \
    --downscale 1 \
    --soft-reset \
    --fc-bottleneck \
    --fc-recurrent \
    --conv-recurrent \


# sleep 5

# sbatch run_snn.sh \
#     --model unet \
#     --data det \
#     --description "U-net Rerun 4" \
#     --features 32 64 128 256 \
#     --initial-scaling 1 \
#     --used-T 30 \
#     --downscale 4 \
#     --not-analog \
#     --soft-reset \
#     --no-fc-bottleneck \
#     --no-fc-recurrent \
#     --no-conv-recurrent \



# sbatch run_snn.sh \
#     --model unet \
#     --data det \
#     --description "Recurrent DET rerun 1" \
#     --features 16 24 32 48 \
#     --epochs 200 \
#     --initial-scaling 4 \
#     --downscale 1 \
#     --soft-reset \
#     --fc-bottleneck \
#     --fc-recurrent \
#     --conv-recurrent \
    
# sleep 5

# sbatch run_snn.sh \
#     --model unet \
#     --data det \
#     --description "Best with static input" \
#     --features 16 24 32 48 \
#     --initial-scaling 4 \
#     --static-data \
#     --downscale 1 \
#     --soft-reset \
#     --fc-bottleneck \
#     --fc-recurrent \
#     --conv-recurrent \
    
# sleep 5

# sbatch run_snn.sh \
#     --model unet \
#     --data det \
#     --description "Best with T = 10" \
#     --features 16 24 32 48 \
#     --initial-scaling 4 \
#     --used-T 10 \
#     --downscale 1 \
#     --soft-reset \
#     --fc-bottleneck \
#     --fc-recurrent \
#     --conv-recurrent \
    


# sbatch run_snn.sh \
#     --model lanesnn \
#     --data det \
#     --description "LaneSNN Rerun 5" \
#     --features 600 \
#     --used-T 30 \
#     --downscale 4 \