#!/usr/bin/env bash
dockerfile=Dockerfile

# Prompt user for image_name, default to masked_mimic:v2
read -p "Enter image tag (default: v2 -> masked_mimic:v2): " image_tag
image_name=masked_mimic:${image_tag:-v2}
docker build --build-arg UID=$UID -t "${image_name}" -f "${dockerfile}" .

# Run the local_package_installs.sh script in the container
docker run --name temp_container --gpus=all -e "ACCEPT_EULA=Y" -v ~/dev/ProtoMotions:/home/gymuser/ProtoMotions --ipc=host "${image_name}" bash -c "docker/local_package_installs.sh"
# This is here because the data_gen_invocation_script.sh is not able to run in the Dockerfile
docker run --name temp_container --gpus=all -e "ACCEPT_EULA=Y" -v ~/dev/ProtoMotions:/home/gymuser/ProtoMotions --ipc=host "${image_name}" bash -c "docker/data_gen_invocation_script.sh"
docker commit temp_container "${image_name}"
docker rm temp_container