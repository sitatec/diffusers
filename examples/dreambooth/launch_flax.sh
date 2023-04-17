export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="~/training_data/subject1"
export OUTPUT_DIR="~/models/subject1"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --train_text_encoder \
  --revision="fp16" \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sbjI" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --sample_batch_size=4 \
  --max_train_steps=4000 \
  --save_interval=400 \
  --save_sample_prompt="a photo of sbjI" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0
  
