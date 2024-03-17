#! /bin/bash

# Function to sync output directory to S
# W A R N I N G: This function will run indefinitely until killed by another process

function shutdown_pod() {
    # Prompt for user confirmation before shutdown
    echo "System will shutdown in 30 seconds. Press any key to abort."

    # Set the time limit in seconds
    TIME_LIMIT=30

    # Read user input with a timeout
    read -t $TIME_LIMIT -n 1 -s -r -p "Press any key to abort..."

    # Check if the user pressed a keyk
    if [ $? -eq 0 ]; then
        echo "Shutdown aborted by user."
        exit 1
    else
        echo "No key pressed. Proceeding with shutdown..."
        runpodctl stop pod $RUNPOD_POD_ID
    fi
}

######### Start Training #########
echo "Starting training"

[[ RUNPOD_GPU_COUNT -gt 1 ]] && MULTI_GPU="--multi_gpu" || MULTI_GPU=""

#Continuously output to S3 durring training
sync_output_to_s3 &
sync_pid=$!

cd /workspace/kohya_ss
source venv/bin/activate
mkdir -p ${LOGGING_DIR}/logs

accelerate launch --num_cpu_threads_per_process=2 \
    $MULTI_GPU \
    --num_processes="${RUNPOD_GPU_COUNT}" \
    --num_machines=1 \
    --config_file=/workspace/training/accelerate.yaml \
    "./sdxl_train_network.py" \
    --enable_bucket \
    --min_bucket_reso=256 \
    --max_bucket_reso=2048 \
    --bucket_reso_steps=64 \
    --cache_latents \
    --cache_latents_to_disk \
    --caption_dropout_rate="0.05" \
    --gradient_checkpointing \
    --learning_rate=8e-5 \
    --text_encoder_Lr=5.2e-5 \
    --lr_scheduler="constant_with_warmup" \
    --lr_scheduler_num_cycles="50" \
    --max_grad_norm="1" \
    --resolution="1024,1024" \
    --max_train_epochs=100 \
    --max_train_steps="10000" \
    --min_snr_gamma=5 \
    --mixed_precision="fp16" \
    --no_half_vae \
    --optimizer_type="Adafactor" \
    --optimizer_args scale_parameter=False relative_step=False warmup_init=False \
    --output_dir="${OUTPUT_DIR}" \
    --output_name="${OUTPUT_MODEL_NAME}" \
    --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --save_every_n_epochs="1" \
    --save_model_as=safetensors \
    --save_precision="fp16" \
    --train_batch_size="${BATCH_SIZE}" \
    --train_data_dir="/workspace/training/dataset" \
    --xformers \
    --sample_sampler=euler_a \
    --sample_prompts="/workspace/training/prompt.txt" \
    --sample_every_n_epochs="1" \
    --sample_every_n_steps="100" \
    --gradient_accumulation_steps=4 \
    --log_with="tensorboard" \
    --logging_dir="workspace/logs" \
    --max_token_length="150" \
    --dataset_repeats=${DATASET_REPEATS} \
    --shuffle_caption \
    --in_json="/workspace/training/captions.json" \
    --save_state

#finish syncing output
# aws s3 sync /workspace/output s3://${S3_OUTPUT_BUCKET}/training-output --recursive

# Will wait for user to interrupt for 30 seconds before shutting down the pod
shutdown_pod
