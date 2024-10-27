#!/usr/bin/env bash

# Prompt user for container_name, default to masked_mimic
read -rp "Enter container_name (default: masked_mimic): " container_name
container_name=${container_name:-masked_mimic}

# Prompt user for ssh_port, default to 8060
read -rp "Enter ssh_port (default: 8060): " ssh_port
ssh_port=${ssh_port:-8060}

# read image tag, default to v2
read -rp "Enter image tag (default: v2 -> masked_mimic:v2): " image_tag
image_name=masked_mimic:${image_tag:-v2}

# Check if nvidia-docker is installed, if not then default to docker
if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

if [ "$display" != "" ]; then
  ${cmd} run --rm --name "${container_name}" \
    -p "${ssh_port}":22 \
    -p "${tensorboard_port}":6006 \
    -v /tmp:/tmp \
    -v ~/dev/MaskedMimic:/home/gymuser/MaskedMimic \
    --gpus all \
    --ipc=host \
    -e HYDRA_FULL_ERROR=1 \
    -it ${image_name} /bin/bash
else
  xhost +local:root
  ${cmd} run --rm --name "${container_name}" \
    -p "${ssh_port}":22 \
    -p "${tensorboard_port}":6006 \
    -v /tmp:/tmp \
    -v ~/.ssh:/root/.ssh \
    -v ~/.ssh:/home/gymuser/.ssh \
    -e DISPLAY="$DISPLAY" \
    -e "ACCEPT_EULA=Y" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="${DISPLAY}" \
    -v ~/dev/MaskedMimic:/home/gymuser/MaskedMimic \
    --gpus all \
    --ipc=host \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    -e HYDRA_FULL_ERROR=1 \
    -it ${image_name} /bin/bash
  xhost -local:root
fi

${cmd} exec "${container_name}" bash