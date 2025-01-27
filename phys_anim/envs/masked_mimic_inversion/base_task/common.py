# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import torch

from isaac_utils import torch_utils, rotations

from phys_anim.utils.motion_lib import MotionLib

from rich.console import Console
from rich.table import Table

from typing import Optional, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic_inversion.base_task.isaacgym import (
        MaskedMimicTaskHumanoid,
    )
else:
    MaskedMimicTaskHumanoid = object


# TODO: heading, pacer path follower, location, reach
#  task defines all parameters, by default they should be fully masked out.
#  allow task to control joints in several forms, e.g., pelvis position in X frames or all frames, etc...
#  see what works best, submit that. Just like PACER/ASE tried multiple rewards/paths until they figured
#  what works best.

# TODO: make sure no reset due to end of motion file, just keep going until the end of the episode.


class BaseMaskedMimicTask(MaskedMimicTaskHumanoid):  # type: ignore[misc]
    def __init__(self, config, device, motion_lib: Optional[MotionLib] = None):
        super().__init__(config, device, motion_lib=motion_lib)
        self.setup_task()

        self.current_pose_obs_type = self.config.get("current_pose_obs_type", None)

        self._text_embedding = None
        if self.config.get("use_text", False):
            self.text_command = self.config.get(
                "text_command", "a person is walking upright"
            )
            text_embedding = get_text_embedding(
                text_command=self.text_command, device=self.device
            )
            self._text_embedding = text_embedding

        self.inversion_obs = torch.zeros(
            (self.config.num_envs, self.config.task_obs_size + self.config.current_pose_obs_size),
            device=device,
            dtype=torch.float,
        )

        self._failures = []
        self._distances = []
        self._current_accumulated_errors = (
                torch.zeros([self.num_envs], device=self.device, dtype=torch.float) - 1
        )
        self._current_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.float
        )
        self._last_length = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )

        self.results = {}
        self.console = Console()

    def accumulate_errors(self):
        self.last_unscaled_rewards = self.log_dict

        if len(self._failures) > 0:
            self.results["reach_success"] = 1.0 - sum(self._failures) / len(
                self._failures
            )
            self.results["reach_distance"] = sum(self._distances) / len(
                self._distances
            )

    def compute_failures_and_distances(self):
        # need to implement this in each env
        pass

    def get_current_pose_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        global_translations = self.get_body_positions()[env_ids]
        root_height = global_translations[env_ids, 0, 2]
        head_height = global_translations[env_ids, self.head_id, 2]
        root_coords = global_translations[env_ids, 0, :]
        head_coords = global_translations[env_ids, self.head_id, :]

        if self.current_pose_obs_type == 'root_head_coords':
            current_pose_obs = torch.cat([root_coords, head_coords], dim=-1)
        elif self.current_pose_obs_type == 'root_head_heights':
            current_pose_obs = torch.cat([root_height.unsqueeze(-1), head_height.unsqueeze(-1)], dim=-1)
        else:
            current_pose_obs = torch.tensor([], device=self.device)
        return current_pose_obs

    ###############################################################
    # Set up environment
    ###############################################################
    def setup_task(self):
        pass

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        if len(env_ids) > 0:
            self.reset_task(env_ids)

    def reset_task(self, env_ids):
        # Make sure in user-control mode that the history isn't visible.
        self.valid_hist_buf.set_all(False)

    ###############################################################
    # Environment step logic
    ###############################################################
    def create_chens_prior(self, env_ids):
        raise NotImplementedError

    def compute_observations(self, env_ids=None):
        self.mask_everything()
        super().compute_observations(env_ids)
        self.mask_everything()
        self.compute_priors(env_ids)

    def compute_priors(self, env_ids):
        if self.config.get("use_chens_prior", False):
            self.create_chens_prior(env_ids)
        if self.config.get("use_text", False):
            self.motion_text_embeddings_mask[:] = True
            self.motion_text_embeddings[:] = self._text_embedding
        if self.config.get("use_raise_hands_prior", False):
            self.create_raise_hands_prior(env_ids)

    def compute_humanoid_obs(self, env_ids=None):
        humanoid_obs = super().compute_humanoid_obs(env_ids)

        # After the humanoid obs is called, we compute the task obs.
        # A task obs will have its own unique tensor.
        # We do not append the task obs to the humanoid obs, rather we allow the user to define the network structure
        # and how the task obs is used in the network.
        self.compute_task_obs()

        return humanoid_obs

    def compute_task_obs(self, env_ids=None):
        self.current_pose_obs = self.get_current_pose_obs(env_ids)

    def mask_everything(self):
        # By Default mask everything out. Individual tasks will override this.
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_bodies_masks[:] = False
        self.target_pose_obs_mask[:] = False
        self.object_bounding_box_obs_mask[:] = False
        self.motion_text_embeddings_mask[:] = False

    def update_task(self, actions):
        pass

    ###############################################################
    # Helpers
    ###############################################################
    def draw_task(self):
        return

    def print_results(self, results_dict: Dict[str, str]) -> None:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="dim")
        table.add_column("Value", justify="right")
        table.add_row("Reward", f"{self.rew_buf.item():.3f}")
        for key, value in results_dict.items():
            # if the value is a float, format it to 3 decimal places
            if isinstance(value, float):
                value = f"{value:.3f}"
            # if the value is a tensor of shape [1], convert it to a float and format it to 3 decimal places
            elif isinstance(value, torch.Tensor) and value.shape == (1,):
                value = f"{value.item():.3f}"
            # if the value is a tensor of bigger shape, convert it to a string but only values, and format them to 3 decimal places
            elif isinstance(value, torch.Tensor):
                # Flatten the tensor to ensure iteration works for multi-dimensional tensors
                value = ", ".join([f"{v.item():.3f}" for v in value.flatten()])

            table.add_row(key, value)

        # Clear the console and print the table
        self.console.clear()
        self.console.print(table)

    def create_raise_hands_prior(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        right_hand_body_index = self.config.masked_mimic_conditionable_bodies.index("R_Hand")
        left_hand_body_index = self.config.masked_mimic_conditionable_bodies.index("L_Hand")
        self.right_hand_body_id = self.build_body_ids_tensor(['R_Hand'])
        self.left_hand_body_id = self.build_body_ids_tensor(['L_Hand'])
        self.target_pose_joints[:] = False
        self.target_pose_joints[:, right_hand_body_index * 2] = True
        self.target_pose_joints[:, left_hand_body_index * 2] = True
        self.target_pose_obs_mask[:] = True

        new_mask = torch.zeros(
            (self.num_envs, self.config.masked_mimic_obs.num_future_steps, self.num_conditionable_bodies, 2),
            dtype=torch.bool,
            device=self.device, )

        new_mask[:, :, right_hand_body_index, 0] = True
        new_mask[:, :, left_hand_body_index, 0] = True

        new_mask = new_mask.reshape(self.num_envs, -1)

        self.masked_mimic_target_bodies_masks[:] = new_mask
        sparse_target_poses = self.build_sparse_target_raise_hands_poses_masked_with_time(
            self.config.masked_mimic_obs.num_future_steps, env_ids
        )
        self.masked_mimic_target_poses[:] = sparse_target_poses
        self.masked_mimic_target_poses_masks[env_ids] = True

    def build_sparse_target_raise_hands_poses_masked_with_time(self, num_future_steps, env_ids=None):
        num_envs = len(env_ids)
        time_offsets = (
                torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
                * self.dt
        )

        near_future_times = self.motion_times[env_ids].unsqueeze(
            -1
        ) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time[env_ids].view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_raise_hands_poses(all_future_times, env_ids).view(
            self.num_envs,
            num_future_steps + 1,
            self.masked_mimic_conditionable_bodies_ids.shape[0] + 1,
            2,
            12,
        )

        near_mask = self.masked_mimic_target_bodies_masks[env_ids].view(
            num_envs, num_future_steps, self.num_conditionable_bodies, 2, 1
        )
        far_mask = self.target_pose_joints[env_ids].view(num_envs, 1, -1, 2, 1)
        mask = torch.cat([near_mask, far_mask], dim=1)

        masked_obs = obs * mask

        masked_obs_with_joints = torch.cat((masked_obs, mask), dim=-1).view(
            num_envs, num_future_steps + 1, -1
        )

        times = all_future_times.view(-1).view(
            num_envs, num_future_steps + 1, 1
        ) - self.motion_times[env_ids].view(num_envs, 1, 1)
        ones_vec = torch.ones(num_envs, num_future_steps + 1, 1, device=self.device)
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(num_envs, -1)

    def build_sparse_target_raise_hands_poses(self, raw_future_times, env_ids):
        """
        This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        num_envs = len(env_ids)
        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos = ref_state.rb_pos
        flat_target_rot = ref_state.rb_rot
        flat_target_vel = ref_state.rb_vel

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = current_state.body_pos[env_ids], current_state.body_rot[env_ids]

        # override to set the target root parameters
        reshaped_target_pos = flat_target_pos.reshape(
            self.num_envs, num_future_steps, -1, 3
        )
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(num_envs, 1, 1)
        # cur_head_pos = cur_gt[:, self.right_hand_body_id]

        expanded_body_pos = cur_gt.unsqueeze(1).expand(
            self.num_envs, num_future_steps, *cur_gt.shape[1:]
        )
        expanded_body_rot = cur_gr.unsqueeze(1).expand(
            self.num_envs, num_future_steps, *cur_gr.shape[1:]
        )

        hands_pos = torch.zeros((num_envs, num_future_steps, 2, 3), device=self.device, dtype=torch.float32)
        hands_dict = {
            "hands_height": 1.2,
            "hands_distance": 0.3,
            "hands_gap": 0.4
        }
        hands_pos[..., 0, 0] = hands_dict["hands_distance"]
        hands_pos[..., 0, 1] = hands_dict["hands_gap"] * 1. / 2
        hands_pos[..., 0, 2] = hands_dict["hands_height"]

        hands_pos[..., 1, 0] = hands_dict["hands_distance"]
        hands_pos[..., 1, 1] = -hands_dict["hands_gap"] * 1. / 2
        hands_pos[..., 1, 2] = hands_dict["hands_height"]

        # Calculate expanded_root_heading properly
        expanded_root_rot = expanded_body_rot[:, :, 0, :]  # Extract root rotation
        expanded_root_heading = torch_utils.calc_heading_quat_inv(expanded_root_rot.reshape(-1, 4), self.w_last)
        expanded_root_heading = expanded_root_heading.reshape(num_envs, num_future_steps, 1, 4).expand(-1, -1, 2,
                                                                                                       -1)  # Repeat for two hands

        # Prepare hands_pos for rotation
        hands_pos_flat = hands_pos.reshape(-1, 3)  # Flatten hands_pos to match expected input shape

        # Apply rotation to hands positions
        rotated_hands_pos = rotations.quat_rotate(expanded_root_heading.reshape(-1, 4), hands_pos_flat, w_last=True)
        rotated_hands_pos = rotated_hands_pos.reshape(num_envs, num_future_steps, 2, 3)  # Reshape back

        # Translate to root position
        root_positions = cur_gt[:, None, :, :]  # [num_envs, 1, num_bodies, 3]
        rotated_hands_pos = rotated_hands_pos.view(num_envs, num_future_steps, 2, 3)
        world_hands_pos = rotated_hands_pos + root_positions[:, :, [self.right_hand_body_id, self.left_hand_body_id], :]

        # Update target positions
        reshaped_target_pos[:, :, self.right_hand_body_id, :] = world_hands_pos[:, :, 0, :].unsqueeze(-2)
        reshaped_target_pos[:, :, self.left_hand_body_id, :] = world_hands_pos[:, :, 1, :].unsqueeze(-2)

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)
        # override to set the target root parameters

        flat_cur_pos = expanded_body_pos.reshape(flat_target_pos.shape)
        flat_cur_rot = expanded_body_rot.reshape(flat_target_rot.shape)

        root_pos = flat_cur_pos[:, 0, :]
        root_rot = flat_cur_rot[:, 0, :]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, flat_cur_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(
            heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
            heading_rot_expand.shape[2],
        )

        root_pos_expand = root_pos.unsqueeze(-2)

        """target"""
        # target body pos   [N, 3xB]
        target_rel_body_pos = flat_target_pos - flat_cur_pos
        flat_target_rel_body_pos = target_rel_body_pos.reshape(
            target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
            target_rel_body_pos.shape[2],
        )
        flat_target_rel_body_pos = torch_utils.quat_rotate(
            flat_heading_rot, flat_target_rel_body_pos, self.w_last
        )

        # target body pos   [N, 3xB]
        flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
            flat_target_pos.shape[0] * flat_target_pos.shape[1],
            flat_target_pos.shape[2],
        )
        flat_target_body_pos = torch_utils.quat_rotate(
            flat_heading_rot, flat_target_body_pos, self.w_last
        )

        # target body rot   [N, 6xB]
        target_rel_body_rot = rotations.quat_mul(
            rotations.quat_conjugate(flat_cur_rot, self.w_last),
            flat_target_rot,
            self.w_last,
        )
        target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(
            target_rel_body_rot.view(-1, 4), self.w_last
        ).view(target_rel_body_rot.shape[0], -1)

        # target body rot   [N, 6xB]
        target_body_rot = rotations.quat_mul(
            heading_rot_expand, flat_target_rot, self.w_last
        )
        target_body_rot_obs = torch_utils.quat_to_tan_norm(
            target_body_rot.view(-1, 4), self.w_last
        ).view(target_rel_body_rot.shape[0], -1)

        padded_flat_target_rel_body_pos = torch.nn.functional.pad(
            flat_target_rel_body_pos, [0, 3], "constant", 0
        )
        sub_sampled_target_rel_body_pos = padded_flat_target_rel_body_pos.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        padded_flat_target_body_pos = torch.nn.functional.pad(
            flat_target_body_pos, [0, 3], "constant", 0
        )
        sub_sampled_target_body_pos = padded_flat_target_body_pos.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        sub_sampled_target_rel_body_rot_obs = target_rel_body_rot_obs.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]
        sub_sampled_target_body_rot_obs = target_body_rot_obs.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        # Heading
        target_heading_rot = torch_utils.calc_heading_quat(
            flat_target_rot[:, 0, :], self.w_last
        )
        target_rel_heading_rot = torch_utils.quat_to_tan_norm(
            rotations.quat_mul(
                heading_rot_expand[:, 0, :], target_heading_rot, self.w_last
            ).view(-1, 4),
            self.w_last,
        ).reshape(self.num_envs, num_future_steps, 1, 6)

        # Velocity
        target_root_vel = flat_target_vel[:, 0, :]
        target_root_vel[..., -1] = 0  # ignore vertical speed
        target_rel_vel = rotations.quat_rotate(
            heading_rot, target_root_vel, self.w_last
        ).reshape(-1, 3)
        padded_target_rel_vel = torch.nn.functional.pad(
            target_rel_vel, [0, 3], "constant", 0
        )
        padded_target_rel_vel = padded_target_rel_vel.reshape(
            self.num_envs, num_future_steps, 1, 6
        )

        heading_and_velocity = torch.cat(
            [
                target_rel_heading_rot,
                target_rel_heading_rot,
                padded_target_rel_vel,
                padded_target_rel_vel,
            ],
            dim=-1,
        )

        # In masked_mimic allow easy re-shape to [batch, time, joint, type (transform/rotate), features]
        obs = torch.cat(
            (
                sub_sampled_target_rel_body_pos,
                sub_sampled_target_body_pos,
                sub_sampled_target_rel_body_rot_obs,
                sub_sampled_target_body_rot_obs,
            ),
            dim=-1,
        )  # [batch, timesteps, joints, 24]
        obs = torch.cat((obs, heading_and_velocity), dim=-2).view(self.num_envs, -1)

        return obs


def get_text_embedding(
        text_command="a person is walking upright",
        device: torch.device = torch.device("cuda:0"),
):
    from transformers import AutoTokenizer, XCLIPTextModel

    model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

    text_command = [text_command]
    with torch.inference_mode():
        inputs = tokenizer(
            text_command, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
        text_embedding = pooled_output[0].to(device)
        return text_embedding
