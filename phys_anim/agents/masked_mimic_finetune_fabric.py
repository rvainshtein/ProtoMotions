from phys_anim.agents.mimic_vae_dagger import MimicVAEDagger
from phys_anim.envs.humanoid.common import Humanoid
from phys_anim.agents.models.actor import ActorFixedSigmaVAE

import torch
from torch import Tensor
from lightning.fabric import Fabric

from typing import Tuple, Dict


class MimicFinetune(MimicVAEDagger):  # TODO inherit from PPO
    env: Humanoid
    actor: ActorFixedSigmaVAE

    def __init__(self, fabric: Fabric, env: Humanoid, config):
        super().__init__(fabric, env, config)

    def setup(self):
        super().setup()
        # actor_state_dict = self.actor.state_dict()
        # if self.config.pre_trained_maskedmimic_path is not None:
        #    pre_trained_masked_mimic_state_dict = torch.load(self.config.pre_trained_maskedmimic_path, map_location=self.device)
        #    for key, value in pre_trained_masked_mimic_state_dict.items():
        #        actor_state_dict[key] = value
        #    self.actor.load_state_dict(actor_state_dict)
        #    for name, param in self.actor.named_parameters():
        #        if name in pre_trained_masked_mimic_state_dict.keys():
        #            param.requires_grad = False
