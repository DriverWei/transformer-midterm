#!/bin/bash
set -e

echo "Starting Transformer (Seq2Seq) training with RPE and Stability Tricks..."
echo "Evaluation will run every 250 steps."

python train.py \
    --d_model 256 \
    --nhead 4 \
    --d_ff 1024 \
    --num_layers 3 \
    --batch_size 32 \
    --lr 3e-4 \
    --epochs 1 \
    --block_size 128 \
    --dropout 0.1 \
    --seed 42 \
    --save_interval 5 \
    \
    --data_dir "./data" \
    --results_dir "./results" \
    --checkpoint_dir "./checkpoints" \
    \
    --use_rpe \
    --use_scheduler \
    --use_grad_clip \
    --grad_clip 1.0 \
    --optimizer_type 'adamw' \
    \
    --eval_step 250

echo "Training finished. Results are in 'results/' and models in 'checkpoints/'."
echo "Check for 4 plot files: train_loss_step.png, val_loss_step.png, learning_rate_step.png, and gradient_norm_step.png"

