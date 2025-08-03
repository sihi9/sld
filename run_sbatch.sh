#!/bin/bash


# EXP5
sbatch run_snn.sh \
    --model unet \
    --data det \
    --description "deep small fully recurrent with data augmentation" \
    --features 8 16 24 32\
    --fc-bottleneck \
    --fc-recurrent \
    --conv-recurrent \
    


