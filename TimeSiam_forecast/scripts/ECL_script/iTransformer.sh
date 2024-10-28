export CUDA_VISIBLE_DEVICES=0
model_name=iTransformer
train_epochs=50
mask_rate=0.25
pretrain_seq_len=96
sampling_range=6
lineage_tokens=1
representation_using=concat

python -u run.py \
    --task_name timesiam \
    --is_training 0 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL \
    --model $model_name \
    --data ECL \
    --features M \
    --seq_len $pretrain_seq_len \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 512 \
    --d_ff 512 \
    --mask_rate $mask_rate \
    --sampling_range $sampling_range \
    --lineage_tokens $lineage_tokens \
    --train_epochs $train_epochs

checkpoint=./outputs/pretrain_checkpoints/ECL/ckpt_best.pth

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name fine_tune \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model $model_name \
        --data ECL \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --d_model 512 \
        --d_ff 512 \
        --learning_rate 0.0005 \
        --lineage_tokens $lineage_tokens \
        --representation_using $representation_using \
        --load_checkpoints $checkpoint \
        --batch_size 16 \
        --head_dropout 0
done

