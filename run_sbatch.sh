#!/bin/bash


# EXP5
sbatch run_snn.sh \
    --model lanesnn \
    --data det \
    --description "LaneSNN with 200 Epochs" \
    #--features 16 32\
    #--fc-bottleneck \
    #--no-fc-recurrent \
    #--no-conv-recurrent \
    


