from phys_anim.agents.mimic_vae_dagger import MimicVAEDagger
from phys_anim.agents.ppo import PPO
from phys_anim.envs.humanoid.common import Humanoid
from phys_anim.agents.models.actor import ActorFixedSigmaVAE

import torch
from torch import Tensor
from lightning.fabric import Fabric

from typing import Tuple, Dict


class MimicFinetunePPO(PPO):
    env: Humanoid
    actor: ActorFixedSigmaVAE

    def __init__(self, fabric: Fabric, env: Humanoid, config):
        super().__init__(fabric, env, config)

    def setup(self):
        super().setup()
        actor_state_dict = self.actor.state_dict()
        if self.config.pre_trained_maskedmimic_path is not None:
            pre_trained_masked_mimic_state_dict = torch.load(
                self.config.pre_trained_maskedmimic_path, map_location=self.device
            )
            pre_trained_actor_state_dict = pre_trained_masked_mimic_state_dict['actor']
            for param_name, param_val in self.actor.state_dict().items():
                pre_trained_param_val = pre_trained_actor_state_dict.get(param_name)
                if pre_trained_param_val is not None:
                    if param_val.shape == pre_trained_param_val.shape:
                        actor_state_dict[param_name] = pre_trained_param_val
                    else:
                        print(f"Shape mismatch: {param_name}")
                        print(f"Actor shape: {param_val.shape}, Pre-trained shape: {pre_trained_param_val.shape}")

            self.actor.load_state_dict(actor_state_dict,
                                       strict=False)  # strict=False to allow loading partial state_dict
            for name, param in self.actor.named_parameters():
                if name in pre_trained_actor_state_dict.keys():
                    param.requires_grad = False
