export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
train_epochs=50
mask_rate=0.25
pretrain_seq_len=96
sampling_range=6
lineage_tokens=2
representation_using=avg

python -u run.py \
    --pretrain_type timesiam \
    --operation pretrain \
    --task regression \
    --is_training 0 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model $model_name \
    --data regression \
    --features M \
    --seq_len $pretrain_seq_len \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --mask_rate $mask_rate \
    --sampling_range $sampling_range \
    --lineage_tokens $lineage_tokens \
    --train_epochs $train_epochs \
    --masked_rule channel_binomial