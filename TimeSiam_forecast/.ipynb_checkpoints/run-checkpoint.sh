#!/bin/bash

task_name=simmtm                    # [timesiam, simmtm, fine_tune]
root_path=./dataset/saidi           # 数据所在目录
data_path=lab_hourly_average.csv    # 数据文件名
model_id=saidi                      # 模型id
model=PatchTST                      # 模型 [PatchTST]
data=saidi                          # [ECL, ETT, Exchange, saidi, Traffic, Weather]
train_epochs=10                     # 训练轮数
features=6                          # 特征数
seq_len=24
label_len=1
pred_len=1

bash scripts/${data}_script/PatchTST.sh \
    --task_name $task_name \
    --root_path $root_path \
    --data_path $data_path \
    --model_id $model_id \
    --model $model \
    --train_epochs $train_epochs \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    # > logs/${model_id}_${model}_${data}_${task_name}.log 2>&1 &