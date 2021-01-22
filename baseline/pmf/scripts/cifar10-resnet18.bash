#!/bin/bash
dt='CIFAR10'
sd=./out
bs=50
lr=0.01
ar='RESNET18'
lf='CROSSENTROPY'
op='ADAM'
lrs=0.2
lri=30000
iters=100000
bts=1.05
ql=2
wd=0.0001


## PMF
mt='PMF'
lr=0.01
lrs=0.2
bts=1.05
op='ADAM'
wd=0.0001

echo python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql
python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql



# Accoring to BiNN paper:
# Learning rate start: 10−2 
# Learning rate decay type: Step
# LR decay interval (iterations): 30k 
# LR-scale: 0.2 
# Optimizer: Adam
# Weight decay: 10−4
# bts: 1.05