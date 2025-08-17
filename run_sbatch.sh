#!/bin/bash


sbatch run_snn.sh \
    --model unet \
    --data det \
    --description "smaller hidden dim" \
    --features 16 24 32 48 \
    --hidden-dim 256 \
    --fc-bottleneck \
    --fc-recurrent \
    --conv-recurrent \
    


