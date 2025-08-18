#!/bin/bash

sbatch run_snn.sh \
    --model unet \
    --data det \
    --description "Small analog U-Net" \
    --features 16 24 32 48 \
    --initial-scaling 4 \
    --downscale 1 \
    --analog \
    --soft-reset \
    --no-fc-bottleneck \
    --no-fc-recurrent \
    --no-conv-recurrent \

# sleep 5

# sbatch run_snn.sh \
#     --model unet \
#     --data det \
#     --description "U-Net with T = 10" \
#     --features 32 64 128 256 \
#     --initial-scaling 4 \
#     --used-T 10 \
#     --downscale 1 \
#     --analog \
#     --soft-reset \
#     --no-fc-bottleneck \
#     --no-fc-recurrent \
#     --no-conv-recurrent \



# sbatch run_snn.sh \
#     --model unet \
#     --data carla \
#     --description "Best Carla" \
#     --features 16 24 32 48 \
#     --epochs 100 \
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
    