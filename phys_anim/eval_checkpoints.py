import glob
import subprocess
from itertools import cycle
from rich.console import Console
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class WandbConfig:
    entity: str = "phys_inversion"
    project: str = "eval_results"


@dataclass
class EvalConfig:
    checkpoint_paths: List[str] = field(default_factory=lambda: ["results/long_jump_pose/last.ckpt"])
    checkpoint_paths_glob: str = ""  # Use this to override and glob paths dynamically
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    more_options: str = ""
    log_eval_results: bool = True
    wandb: WandbConfig = WandbConfig()
    opts: List[str] = field(default_factory=lambda: ["wdb"])
    num_envs: int = 1024
    games_per_env: int = 1


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: DictConfig):
    console = Console()

    # Merge default config with CLI arguments
    default_cfg = OmegaConf.structured(EvalConfig())
    config = OmegaConf.merge(default_cfg, config)

    # Resolve checkpoint paths
    if config.checkpoint_paths_glob:
        checkpoint_paths = glob.glob(config.checkpoint_paths_glob, recursive=False)
    else:
        checkpoint_paths = config.checkpoint_paths

    gpu_cycle = cycle(config.gpu_ids) if len(config.gpu_ids) > 1 else None
    processes = []

    volume_configuration = f' +num_envs={config.num_envs} +algo.config.num_games={config.num_envs * config.games_per_env}'

    for checkpoint in checkpoint_paths:
        gpu_id = next(gpu_cycle) if gpu_cycle else config.gpu_ids[0]
        cmd = (
            f"python phys_anim/eval_agent.py +robot=smpl +backbone=isaacgym +headless=True"
            f" +checkpoint={checkpoint} +device={gpu_id}"
            f" +wandb.wandb_entity={config.wandb.entity} +wandb.wandb_project={config.wandb.project}"
            f" +opt=[{','.join(config.opts)}]"
            f" {volume_configuration} {config.more_options}"
        )

        if config.log_eval_results:
            cmd += " ++algo.config.log_eval_results=True"

        console.print(f"[bold blue]Running:[/bold blue] [italic]{cmd}[/italic]")

        if gpu_cycle:
            processes.append(subprocess.Popen(cmd, shell=True))
        else:
            subprocess.run(cmd, shell=True)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    main()
