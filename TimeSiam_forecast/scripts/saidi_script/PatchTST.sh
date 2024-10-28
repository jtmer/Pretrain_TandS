#!/bin/bash

pretrain_type=timesiam
operation=fine_tune
task=long_term_forecast         
root_path=./dataset/saidi       
data_path=lab_hourly_average.csv   
model_id=saidi                    
model=PatchTST     
train_epochs=50       
num_class=2 
enc_in=6
dec_in=6
c_out=6   
seq_len=24
label_len=1
pred_len=1

export CUDA_VISIBLE_DEVICES=0
mask_rate=0.25
sampling_range=6
lineage_tokens=2
representation_using=avg

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --pretrain_type)
      pretrain_type="$2"
      shift 2
      ;;
    --operation)
        operation="$2"
        shift 2
        ;;
    --task)
      task="$2"
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
    --num_class)
      num_class="$2"
      shift 2
      ;; 
    --enc_in)
        enc_in="$2"
        shift 2
        ;;
    --dec_in)
        dec_in="$2"
        shift 2
        ;;
    --c_out)
        c_out="$2"
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

checkpoint=./outputs/pretrain_checkpoints/${model_id}/ckpt_best.pth

if [ "$pretrain_type" == "simmtm" ]; then
    mask_rate=0.05
    if [ "$operation" == "pretrain" ]; then
        python -u run.py \
            --pretrain_type $pretrain_type \
            --operation $operation \
            --task $task \
            --is_training 0 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id $model_id \
            --model $model \
            --data saidi \
            --features M \
            --seq_len $seq_len \
            --e_layers 2 \
            --d_layers 1 \
            --d_model 64 \
            --d_ff 256 \
            --n_heads 2 \
            --patch_len 3 \
            --stride 3 \
            --enc_in $enc_in \
            --dec_in $dec_in \
            --c_out $c_out \
            --num_class $num_class \
            --batch_size 8 \
            --mask_rate $mask_rate \
            --train_epochs $train_epochs \
            --masked_rule mask_patch 
    elif [ "$operation" == "fine_tune" ]; then
        python -u run.py \
        --pretrain_type $pretrain_type \
        --operation $operation \
        --task $task \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id \
        --model $model \
        --data saidi \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 64 \
        --d_ff 256 \
        --n_heads 2 \
        --patch_len 3 \
        --stride 3 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --num_class $num_class \
        --load_checkpoints $checkpoint \
        --batch_size 8 \
        --head_dropout 0  
    else
        python -u run.py \
        --pretrain_type $pretrain_type \
        --operation $operation \
        --task $task \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id $model_id \
        --model $model \
        --data saidi \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 64 \
        --d_ff 256 \
        --n_heads 2 \
        --patch_len 3 \
        --stride 3 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --num_class $num_class \
        --batch_size 8 \
        --head_dropout 0 
    fi
    
else
    if [ "$operation" == "pretrain" ]; then
        python -u run.py \
            --pretrain_type $pretrain_type \
            --operation $operation \
            --task $task \
            --is_training 0 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id $model_id \
            --model $model \
            --data saidi \
            --features M \
            --seq_len $seq_len \
            --e_layers 1 \
            --d_layers 1 \
            --d_model 64 \
            --d_ff 256 \
            --n_heads 2 \
            --patch_len 3 \
            --stride 3 \
            --enc_in $enc_in \
            --dec_in $dec_in \
            --c_out $c_out \
            --num_class $num_class \
            --batch_size 8 \
            --mask_rate $mask_rate \
            --sampling_range $sampling_range \
            --lineage_tokens $lineage_tokens \
            --train_epochs $train_epochs \
            --masked_rule channel_binomial
    elif [ "$operation" == "fine_tune" ]; then
        python -u run.py \
            --pretrain_type $pretrain_type \
            --operation $operation \
            --task $task \
            --is_training 1 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id $model_id \
            --model $model \
            --data saidi \
            --features M \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --e_layers 1 \
            --d_layers 1 \
            --d_model 64 \
            --d_ff 256 \
            --n_heads 2 \
            --patch_len 3 \
            --stride 3 \
            --enc_in $enc_in \
            --dec_in $dec_in \
            --c_out $c_out \
            --num_class $num_class \
            --lineage_tokens $lineage_tokens \
            --representation_using $representation_using \
            --load_checkpoints $checkpoint \
            --batch_size 8 \
            --head_dropout 0
    else
        python -u run.py \
            --pretrain_type $pretrain_type \
            --operation $operation \
            --task $task \
            --is_training 1 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id $model_id \
            --model $model \
            --data saidi \
            --features M \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --e_layers 1 \
            --d_layers 1 \
            --d_model 64 \
            --d_ff 256 \
            --n_heads 2 \
            --patch_len 3 \
            --stride 3 \
            --enc_in $enc_in \
            --dec_in $dec_in \
            --c_out $c_out \
            --num_class $num_class \
            --lineage_tokens $lineage_tokens \
            --representation_using $representation_using \
            --batch_size 8 \
            --learning_rate 0.001 \
            --head_dropout 0
    fi
fi
