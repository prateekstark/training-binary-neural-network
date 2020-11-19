#!/bin/bash
dt='MNIST'
sd=./out
bs=128
lr=0.001
ar='LENET300'
lf='CROSSENTROPY'
op='ADAM'
lrs=0.2
lri=7000
iters=100000
bts=1.2
ql=2
wd=0


# According to BiNN paper 

# For MNIST:
# lr =: 0.001
# learning rate decay type =: step
# lri =: 7K
# lrs: 0.2
# Optimizer: Adam
# wd: 0
# bts: 1.2
# batchsize = 100
# epochs = 500



mt='PMF'
lr=0.001
lrs=0.2
bts=1.2
op='ADAM'
wd=0

echo python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql
python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql



# Learning rate start: 10−3 
# Learning rate decay type Step Step Step
# LR decay interval (iterations) 7k 30k 30k
# LR-scale 0.2 0.2 0.2
# Optimizer Adam Adam Adam
# Weight decay 0 10−4 10−4
# ρ 1.2 1.05 1.05