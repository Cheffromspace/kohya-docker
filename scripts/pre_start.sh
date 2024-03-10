#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export APP="kohya_ss"
DOCKER_IMAGE_VERSION_FILE="/workspace/${APP}/docker_image_version"
        echo 'updating kojya_ss'

echo "Template version: ${TEMPLATE_VERSION}"

if [[ -e ${DOCKER_IMAGE_VERSION_FILE} ]]; then
    EXISTING_VERSION=$(cat ${DOCKER_IMAGE_VERSION_FILE})
else
    EXISTING_VERSION="0.0.0"
fi

sync_apps() {
    # Sync application to workspace to support Network volumes
    echo "Syncing ${APP} to workspace, please wait..."
    rsync --remove-source-files -rlptDu /${APP}/ /workspace/${APP}/
    rm -rf /kohya_ss

    echo "${TEMPLATE_VERSION}" > ${DOCKER_IMAGE_VERSION_FILE}
}

fix_venvs() {
    # Fix the venv to make it work from /workspace
    echo "Fixing venv..."
    /fix_venv.sh /kohya_ss/venv /workspace/kohya_ss/venv
}

link_models() {
    # Link model
    ln -s /sd-models/sd_xl_base_1.0.safetensors /workspace/sd_xl_base_1.0.safetensors
}


if [ "$(printf '%s\n' "$EXISTING_VERSION" "$TEMPLATE_VERSION" | sort -V | head -n 1)" = "$EXISTING_VERSION" ]; then
    if [ "$EXISTING_VERSION" != "$TEMPLATE_VERSION" ]; then
        sync_apps
        fix_venvs
        link_models


        # Configure accelerate
        echo "Configuring accelerate..."
        mkdir -p /root/.cache/huggingface/accelerate
        mv /accelerate.yaml /root/.cache/huggingface/accelerate/default_config.yaml
    else
        echo "Existing version is the same as the template version, no syncing required."
    fi
else
    echo "Existing version is newer than the template version, not syncing!"
fi

if [[ ${DISABLE_AUTOLAUNCH} ]]
then
    echo "Auto launching is disabled so the application will not be started automatically"
    echo "You can launch them it using the launcher script:"
    echo ""
    echo "   /start_kohya_ss.sh"
else
    /start_kohya_ss.sh
fi

echo "All services have been started"
