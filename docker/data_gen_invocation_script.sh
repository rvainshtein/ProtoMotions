MOTION_FILE_PATH=phys_anim/data/motions/smpl_demo_motion.npy # This is small dummy data
MODEL_PATH=masked_mimic/masked_mimic_new

echo "Running data gen script for train"
python phys_anim/eval_agent.py \
  +exp=masked_mimic \
  +backbone=isaacgym \
  +robot=smpl \
  +motion_file=${MOTION_FILE_PATH} \
  +checkpoint=${MODEL_PATH}/full_body_tracker/last.ckpt \
  scene_file='data/yaml_files/samp_scenes_train.yaml' \
  +headless=True \
  +ngpu=1 \
  +force_flat_terrain=False

echo "Running data gen script for test"docker
python phys_anim/eval_agent.py \
  +exp=masked_mimic \
  +backbone=isaacgym \
  +robot=smpl \
  +motion_file=${MOTION_FILE_PATH} \
  +checkpoint=${MODEL_PATH}/full_body_tracker/last.ckpt \
  scene_file='data/yaml_files/samp_scenes_test.yaml' \
  +headless=True \
  +ngpu=1 \
  +force_flat_terrain=False