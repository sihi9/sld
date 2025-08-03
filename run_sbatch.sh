#!/bin/bash


# EXP5
sbatch run_snn.sh \
    --model unet \
    --data det \
    --description "deep small non-recurrent unet with small tau and initial scaling" \
    --features 8 16 32 48\
    --fc-bottleneck \
    --no-fc-recurrent \
    --no-conv-recurrent \
    


