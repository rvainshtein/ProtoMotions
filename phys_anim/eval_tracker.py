import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

has_robot_arg = False
backbone = None
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    if "robot" in arg:
        has_robot_arg = True
    if "backbone" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +backbone")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            backbone = "isaacgym"
        elif "isaacsim" in arg.split("=")[-1]:
            from isaacsim import SimulationApp

            backbone = "isaacsim"

from lightning.fabric import Fabric  # noqa: E402
from utils.config_utils import *  # noqa: E402, F403

from phys_anim.agents.mimic import Mimic  # noqa: E402


@hydra.main(config_path="config")
def main(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True

        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                print(f"Could not find config path: {config_path}")

        if has_config:
            print(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if backbone == "isaacsim":
        simulation_app = SimulationApp({"headless": config.headless})

    env = instantiate(config.env, device=fabric.device)

    algo: Mimic = instantiate(config.algo, env=env, fabric=fabric)
    algo.setup()
    algo.load(config.checkpoint)

    algo.evaluate_mimic()


if __name__ == "__main__":
    main()