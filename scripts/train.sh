export MODEL_NAME="/data2404/zhangchi/models/stable-diffusion-v1-4"
export DATASET_NAME="/data2404/zhangchi/dataset/fundus/fundus_data.csv"
export CUDA_VISIBLE_DEVICES="7"
export WANDB_MODE="offline"

accelerate launch --num_processes=1 --mixed_precision="fp16" ../train/train_model.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=20 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --validation_prompts "fundus"\
  --validation_epochs=1 \
  --output_dir="checkpoint/sd-model-finetuned-on-fundus/" \
  --report_to="wandb" \