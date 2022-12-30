export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/workspace/training_data/subject1"
export OUTPUT_DIR="/workspace/models/subject1"
export CLASS_DIR="/workspace/regularisations"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=4 \
  --max_train_steps=2000 \
  --save_interval=400 \
  --save_sample_prompt="a photo of sbjI person" \
  --class_prompt="a photo of person" \
