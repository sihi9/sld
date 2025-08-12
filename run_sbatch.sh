#!/bin/bash


# EXP5
sbatch run_snn.sh \
    --model unet \
    --data carla \
    --description "deep medium full recurrent" \
    --features 16 24 32 48\
    --fc-bottleneck \
    --fc-recurrent \
    --conv-recurrent \
    


