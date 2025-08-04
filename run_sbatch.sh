#!/bin/bash


# EXP5
sbatch run_snn.sh \
    --model unet \
    --data det \
    --description "deep small non recurrent with inital scaling" \
    --features 8 16 24 32\
    --fc-bottleneck \
    --no-fc-recurrent \
    --no-conv-recurrent \
    


