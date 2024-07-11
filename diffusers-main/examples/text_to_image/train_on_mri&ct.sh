export MODEL_NAME="/data2404/zhangchi/models/stable-diffusion-v1-4"  # 这里放入pretrain model, 可以是roentgen，也可以是stable diffusion官方的模型
export DATASET_NAME="/data2404/zhangchi/dataset/0706data.csv" # 这里是一个csv，中间有两列，分别是image路径和对应的text
export CUDA_VISIBLE_DEVICES="0,4,7" # 训练使用的GPU
export WANDB_MODE="offline"

accelerate launch --num_processes=3 --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=20 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --validation_prompts "chest CT,EGFR Sensitive Type,Stage-1" "breast MRI,with tumor,her2,T1C" "brain MRI,glioma,T1,MET" \  # 可以放一些测试的prompts, 修改train_text_to_image.py的第165行内容
  --validation_epochs=1 \
  --output_dir="SD-on-mri&ct" \
  --report_to="wandb" \
  # --resume_from_checkpoint="/home/luyx/EGFR-SD/diffusers-main/examples/text_to_image/sd-EGFR-model/checkpoint-400" # 选择从哪一个checkpoint开始继续训练