# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import torch

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
            #if the value is a tensor of shape [1], convert it to a float and format it to 3 decimal places
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
