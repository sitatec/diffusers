export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="/home/sita/training_data/subject1"
export OUTPUT_DIR="/home/sita/models/subject1"

python3 train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sbjI" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --sample_batch_size=4 \
  --max_train_steps=4000
