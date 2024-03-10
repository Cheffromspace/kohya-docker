#! /bin/bash

mkdir -p /workspace/output/logs

echo "Starting training"
cd /workspace/kohya_ss
source venv/bin/activate
pip install -r requirements.txt
pip install TensorRT

accelerate launch --num_cpu_threads_per_process=2 \
    --config_file=/workspace/training/accelerate.yaml \
    "./sdxl_train.py" \
    --bucket_reso_steps=64 \
    --cache_latents \
    --cache_latents_to_disk \
    --caption_dropout_rate="0.05" \
    --enable_bucket \
    --min_bucket_reso=256 \
    --max_bucket_reso=2048 \
    --gradient_checkpointing \
    --learning_rate=8e-6 \
    --learning_rate_te1=5.2e-6 \
    --learning_rate_te2=4.8e-6 \
    --lr_scheduler="constant_with_warmup" \
    --lr_scheduler_num_cycles="50" \
    --max_grad_norm="1" \
    --resolution="1024,1024" \
    --max_train_epochs=50 \
    --max_train_steps="10000" \
    --min_snr_gamma=5 \
    --mixed_precision="fp16" \
    --no_half_vae \
    --optimizer_args scale_parameter=False relative_step=False warmup_init=False \
    --optimizer_type="Adafactor" \
    --output_dir="/workspace/output" \
    --output_name="trained" \
    --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --save_every_n_epochs="1" \
    --save_model_as=safetensors \
    --save_precision="fp16" \
    --train_batch_size="16" \
    --train_data_dir="/workspace/training/dataset" \
    --xformers \
    --sample_sampler=euler_a \
    --sample_prompts="/workspace/training/prompt.txt" \
    --sample_every_n_epochs="1" \
    --sample_every_n_steps="100" \
    --in_json="/workspace/training/captions.json" \
    --gradient_accumulation_steps=4 \
    --log_with="tensorboard" \
    --logging_dir="/workspace/output/logs" \
    --max_token_length="150" \
    --dataset_repeats="4"

# Export models to s3
aws s3 cp /workspace/output s3://training-output-${S3_BUCKET_NAME} --recursive
# Shut down
exit
