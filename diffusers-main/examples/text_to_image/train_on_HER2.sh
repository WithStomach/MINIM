export MODEL_NAME="/data2404/zhangchi/models/stable-diffusion-v1-4"  # 这里放入pretrain model, 可以是roentgen，也可以是stable diffusion官方的模型
export DATASET_NAME="/data2404/zhangchi/dataset/HER2/her2_metadata.csv" # 这里是一个csv，中间有两列，分别是image路径和对应的text
export CUDA_VISIBLE_DEVICES="1,2,3,5" # 训练使用的GPU
export WANDB_MODE="offline"

accelerate launch --num_processes=4 --mixed_precision="fp16" train_text_to_image.py \
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
  --validation_prompts "breast MRI,with tumor,her2 wild,T1C" "breast MRI,with tumor,her2,T2" "breast MRI,with tumor,her2 wild,T1"\  # 可以放一些测试的prompts, 修改train_text_to_image.py的第165行内容
  --validation_epochs=1 \
  --output_dir="SD-on-MRI" \
  --report_to="wandb" \
  # --resume_from_checkpoint="/data2404/zhangchi/checkpoint/sd-model-finetuned-MET-0609/checkpoint-3300" # 选择从哪一个checkpoint开始继续训练