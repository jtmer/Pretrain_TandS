#!/bin/bash

pretrain_type=timesiam              # [timesiam, simmtm]
operation=fine_tune                 # [pretrain, fine_tune, train]
task=long_term_forecast             # [long_term_forecast]
root_path=./dataset/gas             # 数据所在目录
data_path=train.csv                 # 数据文件名
model_id=gas                        # 模型id
model=PatchTST                      # 模型 [PatchTST]
data=gas                            # [gas, Weather, ...]

train_epochs=10                     # 训练轮数
enc_in=1                            
dec_in=1
c_out=1                    
seq_len=48
label_len=24
pred_len=12

# model_id=${pretrain_type}_${model}_${data}_${features}_${seq_len}_${label_len}_${model_name}

bash scripts/${data}_script/PatchTST.sh \
    --pretrain_type $pretrain_type \
    --operation $operation \
    --task $task \
    --root_path $root_path \
    --data_path $data_path \
    --model_id $model_id \
    --model $model \
    --train_epochs $train_epochs \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    # > logs/${model_id}_${model}_${data}_${task_name}.log 2>&1 &
