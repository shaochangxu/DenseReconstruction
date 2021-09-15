#!/bin/bash
data=$(date +"%m%d")
n=1
batch=1
epochs=10
d=64
interval_scale=1.06
lr=0.001
lr_scheduler=cosinedecay
loss=unsup_loss
optimizer=Adam
loss_w=4
image_scale=0.25
view_num=5

name=${date}_checkpoints
now=$(date +"%Y%m%d_%H%M%S")
echo $name
echo $now

data_set="dtu_yao"
train_path="/disk2/scx/buaa/mvs_training/dtu"
list_file="./lists/dtu/train.txt"

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=$n --master_port 10190 train.py  \
        --model_version=V3 \
        --loss=${loss} \
		--max_h=128 \
		--max_w=160 \
        --dataset=$data_set \
        --trainpath=${train_path} \
        --trainlist=${list_file} \
        --ngpu=$n \
        --lr=$lr \
        --epochs=$epochs \
        --batch_size=$batch \
        --loss_w=$loss_w \
        --using_apex \
        --lr_scheduler=$lr_scheduler \
        --optimizer=$optimizer \
        --view_num=$view_num \
        --image_scale=$image_scale \
        --reg_loss=True \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --logdir=./logdir/${name} \
        --save_dir=./checkpoints \
        #2>&1|tee ./${name}-${now}.log &
