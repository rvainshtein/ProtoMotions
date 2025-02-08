# Our goal in this script is to take  list of checkpoint paths on their envs.
# We use the base run command from eval_agent.py
# We then iterate over the list of checkpoint paths and create a run command for each checkpoint path.
# If given several GPU ids, we can distribute the checkpoints over the GPUs.
# In the end we aggregate all results into a single file, but also provide an option to log to wandb.

import subprocess
from itertools import cycle

from rich.console import Console


def main():
    # Set parameters manually
    checkpoint_paths = ["results/direction_facing_rel_speed/last.ckpt", ]  # List of checkpoint paths
    gpu_ids = [0]  # List of GPU IDs to distribute runs
    more_options = ""  # Additional options for the eval script
    log_eval_results = True  # Whether to set algo.config.log_eval_results
    wandb_project = "eval_results"
    opts = ['wdb']

    num_envs = 1024
    max_games = num_envs * 2

    volume_configuration = f' +num_envs={num_envs} +algo.config.num_games={max_games}'

    gpu_cycle = cycle(gpu_ids) if len(gpu_ids) > 1 else None
    processes = []
    for checkpoint in checkpoint_paths:
        gpu_id = next(gpu_cycle) if gpu_cycle else gpu_ids[0]
        cmd = (
            f"python phys_anim/eval_agent.py +robot=smpl +backbone=isaacgym +headless=True"
            f" +checkpoint={checkpoint} +device={gpu_id}"
            f" +wandb.wandb_entity=phys_inversion +wandb.wandb_project={wandb_project}"
            f" +opt=[{','.join(opts)}]"
            f" {volume_configuration} {more_options}"
        )
        if log_eval_results:
            cmd += " ++algo.config.log_eval_results=True"

        console = Console()
        console.print(f"[bold blue]Running:[/bold blue] [italic]{cmd}[/italic]")

        if gpu_cycle:
            processes.append(subprocess.Popen(cmd, shell=True))
        else:
            subprocess.run(cmd, shell=True)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    main()
