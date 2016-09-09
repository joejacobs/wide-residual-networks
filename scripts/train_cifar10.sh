#!/usr/bin/env bash

export learningRate=0.1
export epoch_step="{60,120,160}"
export max_epoch=200
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection

export dataset=${HOME}/PhD/Data/Public/CIFAR/10/cifar-10.proc.t7
export save=${HOME}/PhD/Data/Public/CIFAR/10/Output/${model}_${RANDOM}${RANDOM}
export batchSize=128

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
mkdir -p $save
th train.lua | tee $save/log.txt
