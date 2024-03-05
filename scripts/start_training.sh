#! /bin/bash

mkdir -p /workspace/output/sample
mv /workspace/training/prompt.txt /workspace/output/sample/prompt.txt

cd /workspace/kohya_ss || exit

accelerate launch --num_cpu_threads_per_process=2 \
"./sdxl_train_network.py" \
--bucket_reso_steps=64 \
--cache_latents \
--cache_latents_to_disk \
--caption_dropout_rate="0.05" \
--enable_bucket \
--min_bucket_reso=256 \
--max_bucket_reso=2048 \
--gradient_checkpointing \
--learning_rate="3e-05" \
--lr_scheduler="constant_with_warmup" \
--lr_scheduler_num_cycles="50" \
--max_data_loader_n_workers="0" \
--max_grad_norm="1" \
--resolution="1024,1024" \
--max_train_epochs=50 \
--max_train_steps="10000" \
--min_snr_gamma=5 \
--mixed_precision="fp16" \
--network_alpha="32" \
--network_dim=32 \
--network_module=networks.lora \
--no_half_vae \
--optimizer_args scale_parameter=False relative_step=False warmup_init=False \
--optimizer_type="Adafactor" \
--output_dir="/workspace/output" \
--output_name="trained" \
--pretrained_model_name_or_path="/workspace/training/animeboysxl_v20.safetensors" \
--save_every_n_epochs="1" \
--save_model_as=safetensors \
--save_precision="fp16" \
--text_encoder_lr=3e-05 \
--train_batch_size="8" \
--training_comment="3 repeats" \
--train_data_dir="/workspace/training/dataset" \
--unet_lr=3e-05 \
--xformers \
--sample_sampler=euler_a \
--sample_prompts="/workspace/output/sample/prompt.txt" \
--sample_every_n_epochs="1" \
--sample_every_n_steps="100" \
--caption_extension="txt"


