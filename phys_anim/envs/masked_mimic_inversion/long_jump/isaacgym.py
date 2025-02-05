from typing import Optional, Tuple, Dict

from isaac_utils.torch_utils import *
from isaac_utils import torch_utils
from torch import Tensor

from phys_anim.envs.masked_mimic_inversion.base_task.isaacgym import MaskedMimicTaskHumanoid

TAR_ACTOR_ID = 1


class MaskedMimicLongJumpHumanoid(MaskedMimicTaskHumanoid):
    def __init__(self, config, device: torch.device, motion_lib: Optional[torch.Tensor] = None):
        super().__init__(config, device)
        self._tar_dist_min = self.config.long_jump_params.tar_dist_min
        self._tar_dist_max = self.config.long_jump_params.tar_dist_max
        self._near_dist = self.config.long_jump_params.near_dist
        self._near_prob = self.config.long_jump_params.near_prob
        self.first_in = True

        self.goal = torch.tensor([30, 0, 1]).to(self.device)
        self.jump_start = 20
        self.tar_speed = 4  # not used?

        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

    def draw_task(self):
        if self.first_in:
            self.first_in = False
            self.gym.clear_lines(self.viewer)
            cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            for i, env_ptr in enumerate(self.envs):
                vertices = np.array([
                    [0, -0.5, 0],
                    [0, 0.5, 0],
                    [self.jump_start, 0.5, 0],
                    [self.jump_start, -0.5, 0]
                ], dtype=np.float32)

                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]]
                ])
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)

                vertices = np.array([
                    [self.jump_start, -1.5, 0],
                    [self.jump_start, 1.5, 0],
                    [self.goal[0], 1.5, 0],
                    [self.goal[0], -1.5, 0]
                ], dtype=np.float32)

                lines = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]]
                ])
                for line in lines:
                    self.gym.add_lines(self.viewer, env_ptr, 1, line, cols)

    def compute_task_obs(self, env_ids=None):
        root_states = self.get_humanoid_root_states()
        obs = compute_longjump_observations(root_states, self.goal, self.jump_start, self.w_last)

        return obs

    def compute_reward(self, actions):
        # reward = 1
        root_states = self.root_states
        self.rew_buf[:], output_dict = compute_longjump_reward(root_states,
                                                  self._prev_root_pos,
                                                  self.goal,
                                                  self.jump_start,
                                                  self.rigid_body_pos,
                                                  self.contact_forces,
                                                  self.contact_body_ids)

        self._prev_root_pos[:] = root_states[:, 0:3].clone()

        self.log_dict.update(output_dict)
        # # need these at the end of every compute_reward function
        # self.compute_failures_and_distances()
        # self.accumulate_errors()

    def compute_reset(self):

        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(self.reset_buf,
                                                                          self.progress_buf,
                                                                          self.contact_forces,
                                                                          self.contact_body_ids,
                                                                          self.rigid_body_pos,
                                                                          self.config.max_episode_length,
                                                                          self.config.enable_height_termination,
                                                                          self.termination_heights,
                                                                          self.jump_start)


#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_longjump_observations(root_states, goal, jump_start, w_last):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot, w_last)
    local_tar_pos = goal - root_pos
    local_tar_pos = torch_utils.quat_rotate(heading_rot_inv, local_tar_pos, w_last)

    humanoid_jumpstart_diff = jump_start - root_pos[:, 0: 1]
    obs = torch.concatenate((local_tar_pos, humanoid_jumpstart_diff), dim=-1)
    return obs


@torch.jit.script
def compute_longjump_reward(root_states, prev_root_pos, goal, jump_start, rigid_body_pos, contact_buf,
                            contact_body_ids):
    # type: (Tensor, Tensor, Tensor, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Dict[str, Tensor]]
    root_pos = root_states[:, 0:3]
    prev_dist = torch.norm(prev_root_pos - goal, dim=-1)
    curr_dist = torch.norm(root_pos - goal, dim=-1)
    closer_target_r = torch.clamp(prev_dist - curr_dist, min=0, max=1)

    # root velocity in x reward
    vel_reward = root_states[:, 7]

    # jump height reward
    x_over_40 = torch.any(rigid_body_pos[:, contact_body_ids, 0] > jump_start, dim=-1)  # shape 1024
    jump_height_reward = torch.zeros(root_states.shape[0]).to(root_states.device)
    jump_height_reward[x_over_40] = root_states[x_over_40, 2]

    # end reward for jump length
    jump_length_reward = torch.zeros(root_states.shape[0]).to(root_states.device)
    force_threshold = 50
    contact_force_not_zero = torch.sqrt(
        torch.sum(torch.sum(torch.square(contact_buf), dim=-1), dim=-1)) > force_threshold
    reset_x_over_40_and_contact_force_not_zero = torch.logical_and(x_over_40, contact_force_not_zero)

    jump_length = torch.mean(rigid_body_pos[reset_x_over_40_and_contact_force_not_zero][:, :, 0],
                             dim=-1) - jump_start

    jump_length_reward[reset_x_over_40_and_contact_force_not_zero] = torch.mean(
        rigid_body_pos[reset_x_over_40_and_contact_force_not_zero][:, :, 0], dim=-1) - jump_start

    # parameters
    closer_target_r *= 1
    vel_reward *= 0.01
    jump_height_reward *= 0.1
    jump_length_reward *= 30
    reward = closer_target_r + vel_reward + jump_height_reward + jump_length_reward
    # print("closer_target_r", closer_target_r[0], "vel", vel_reward[0], "height", jump_height_reward[0], "length", jump_length_reward[0], "total", reward[0])

    output_dict = {
        "closer_target_r": closer_target_r,
        "vel_reward": vel_reward,
        "jump_height_reward": jump_height_reward,
        "jump_length_reward": jump_length_reward,
        "jump_length": jump_length,
    }
    return reward, output_dict


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, jump_start):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    force_threshold = 50
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:

        # ------------contact_buf_list[0].shape
        # torch.Size([1024, 24, 3])
        # ----------(Pdb) rigid_body_pos_list[0].shape
        # torch.Size([1024, 24, 3])
        # ----------(Pdb) contact_body_ids
        # tensor([7, 3, 8, 4], device='cuda:0')
        # -----------reset_buf.shape
        # torch.Size([1024])
        # -----------progress_buf.shape
        # torch.Size([1024])
        # ------------termination_heights
        # tensor([0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500,
        # 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500,
        # 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.3000], device='cuda:0')

        x_over_40 = torch.any(rigid_body_pos[:, contact_body_ids, 0] > jump_start, dim=-1)  # shape 1024

        contact_force_not_zero = torch.sqrt(
            torch.sum(torch.sum(torch.square(contact_buf[0]), dim=-1), dim=-1)) > force_threshold

        reset_x_over_40_and_contact_force_not_zero = torch.logical_and(x_over_40, contact_force_not_zero)

        jump_length = torch.mean(rigid_body_pos[reset_x_over_40_and_contact_force_not_zero][:, :, 0],
                                 dim=-1) - jump_start
        # if not torch.equal(jump_length, torch.tensor([]).to(jump_length.device)):
        #     print("jump length is ", jump_length)

        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        body_y = rigid_body_pos[..., 0, 1]
        body_out = torch.abs(body_y) > 0.5

        has_fallen = torch.logical_or(fall_contact, fall_height)  # don't touch the hurdle.
        has_fallen = torch.logical_or(has_fallen, body_out)
        has_fallen = torch.logical_or(has_fallen, reset_x_over_40_and_contact_force_not_zero)

        has_failed = has_fallen
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
