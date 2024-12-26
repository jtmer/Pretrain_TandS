export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
train_epochs=50
mask_rate=0.25
pretrain_seq_len=96
sampling_range=6
lineage_tokens=2
representation_using=avg


checkpoint=./outputs/pretrain_checkpoints/regression/ckpt_best.pth

python -u run.py \
    --pretrain_type timesiam \
    --operation fine_tune \
    --task regression \
    --is_training 1 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh2.csv \
    --model_id ETTh2 \
    --model $model_name \
    --data regression \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --lineage_tokens $lineage_tokens \
    --representation_using $representation_using \
    --head_dropout 0.3 \
    --load_checkpoints $checkpoint \