import torch
from isaac_utils import torch_utils, rotations
from isaac_utils.rotations import quat_rotate
from isaac_utils.torch_utils import calc_heading_quat
from typing import TYPE_CHECKING

from phys_anim.envs.masked_mimic_inversion.steering.common import compute_heading_reward

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic_inversion.direction_facing.isaacgym import MaskedMimicDirectionFacingHumanoid

else:
    MaskedMimicDirectionFacingHumanoid = object


class MaskedMimicBaseDirectionFacing(MaskedMimicDirectionFacingHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        self.inversion_obs = torch.zeros(
            (config.num_envs, config.steering_params.obs_size + 2 + self.pose_obs_size),
            device=device,
            dtype=torch.float
        )
        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

    def compute_task_obs(self, env_ids=None):
        super().compute_task_obs(env_ids)
        if env_ids is None:
            root_states = self.get_humanoid_root_states()
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
        facing_obs = compute_facing_observations(root_states, self._tar_dir, self.w_last)
        self.inversion_obs = torch.cat([self.direction_obs, facing_obs], dim=-1)


@torch.jit.script
def compute_facing_observations(root_states, tar_face_dir, w_last: bool):
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    tar_face_dir3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)
    local_tar_face_dir = rotations.quat_rotate(heading_rot, tar_face_dir3d, w_last)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]
    return local_tar_face_dir


# @torch.jit.script
def compute_facing_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, tar_face_dir, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    dir_reward, output_dict = compute_heading_reward(root_pos, prev_root_pos, tar_dir, tar_speed, dt)

    dir_reward_w = 0.7
    facing_reward_w = 0.3
    heading_rot = calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward
