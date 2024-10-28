#!/bin/bash

task_name=timesiam             
root_path=./dataset/ETT-small/      
data_path=ETTh1.csv  
model_id=ETTh1                    
model=PatchTST     
train_epochs=50        
features=7   
seq_len=96
label_len=48
pred_len=96

export CUDA_VISIBLE_DEVICES=0
mask_rate=0.25
sampling_range=6
lineage_tokens=2
representation_using=avg
checkpoint=./outputs/pretrain_checkpoints/ETTh1/ckpt_best.pth

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --task_name)
      task_name="$2"
      shift 2
      ;;
    --root_path)
      root_path="$2"
      shift 2
      ;;
    --data_path)
      data_path="$2"
      shift 2
      ;;
    --model_id)
      model_id="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --train_epochs)
      train_epochs="$2"
      shift 2
      ;;
    --features)
      features="$2"
      shift 2
      ;; 
    --seq_len)
      seq_len="$2"
      shift 2
      ;;
    --label_len)
      label_len="$2"
      shift 2
      ;;
    --pred_len)
      pred_len="$2"
      shift 2
      ;;   
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ "$task_name" == "simmtm" ]; then
    mask_rate=0.05
    python -u run.py \
        --task_name $task_name \
        --is_training 0 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id \
        --model $model \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --e_layers 1 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 1024 \
        --enc_in $features \
        --dec_in $features \
        --c_out $features \
        --mask_rate $mask_rate \
        --train_epochs $train_epochs \

    python -u run.py \
        --task_name fine_tune \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id \
        --model $model \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 1024 \
        --factor 3 \
        --enc_in $features \
        --dec_in $features \
        --c_out $features \
        --load_checkpoints $checkpoint \
        --head_dropout 0.2 
else
    python -u run.py \
        --task_name $task_name \
        --is_training 0 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id \
        --model $model \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --e_layers 1 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 1024 \
        --enc_in $features \
        --dec_in $features \
        --c_out $features \
        --mask_rate $mask_rate \
        --sampling_range $sampling_range \
        --lineage_tokens $lineage_tokens \
        --train_epochs $train_epochs \

    python -u run.py \
        --task_name fine_tune \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id \
        --model $model \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 1024 \
        --factor 3 \
        --enc_in $features \
        --dec_in $features \
        --c_out $features \
        --lineage_tokens $lineage_tokens \
        --representation_using $representation_using \
        --load_checkpoints $checkpoint \
        --head_dropout 0.2
fi

# python -u run.py \
#     --task_name timesiam \
#     --is_training 0 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len $pretrain_seq_len \
#     --e_layers 1 \
#     --d_layers 1 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 512 \
#     --d_ff 1024 \
#     --mask_rate $mask_rate \
#     --sampling_range $sampling_range \
#     --lineage_tokens $lineage_tokens \
#     --train_epochs $train_epochs

# checkpoint=./outputs/pretrain_checkpoints/ETTh1/ckpt_best.pth

# for pred_len in 96 192 336 720; do
#     python -u run.py \
#         --task_name fine_tune \
#         --is_training 1 \
#         --root_path ./dataset/ETT-small/ \
#         --data_path ETTh1.csv \
#         --model_id ETTh1 \
#         --model $model_name \
#         --data ETTh1 \
#         --features M \
#         --seq_len 96 \
#         --label_len 48 \
#         --pred_len $pred_len \
#         --e_layers 1 \
#         --d_layers 1 \
#         --d_model 512 \
#         --d_ff 1024 \
#         --factor 3 \
#         --enc_in 7 \
#         --dec_in 7 \
#         --c_out 7 \
#         --lineage_tokens $lineage_tokens \
#         --representation_using $representation_using \
#         --load_checkpoints $checkpoint \
#         --head_dropout 0.2
# done
