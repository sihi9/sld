#!/bin/bash


sbatch run_snn.sh \
    --model unet \
    --data det \
    --description "default config analog" \
    --features 16 24 32 48\
    --analog \
    --fc-bottleneck \
    --fc-recurrent \
    --conv-recurrent \
    


