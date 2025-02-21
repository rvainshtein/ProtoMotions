import glob
import os
import subprocess
from typing import List


def get_checkpoints_arg(base_dir, env_name, perturbation, prior_only, record_video):
    if prior_only:
        checkpoints_arg = glob.glob(f"{base_dir}/{env_name}/*"
                                    f"_prior_True_text_False_current_pose_True_bigger_True_train_actor_False_0/"
                                    f"last.ckpt")[0]
        checkpoints_arg = f"+checkpoint_paths=[{checkpoints_arg}]"
    else:
        if env_name != 'reach' or (env_name == 'reach' and perturbation == "None"):
            if record_video:
                checkpoints_arg = f"+checkpoint_paths_glob={base_dir}/{env_name}/*_0/last.ckpt"
            else:
                checkpoints_arg = f"+checkpoint_paths_glob={base_dir}/{env_name}/*/last.ckpt"
        else:
            reach_runs = get_valid_reach_runs(base_dir, f'reach/*/last.ckpt',
                                              f'inversion_direction_facing/*/last.ckpt')
            if record_video:
                reach_runs = [run + '/last.ckpt' for run in reach_runs if '_0/' in run]
            else:
                reach_runs = [run + '/last.ckpt' for run in reach_runs]
            checkpoints_arg = f"+checkpoint_paths=[{','.join(reach_runs)}]"
    return checkpoints_arg


def get_valid_reach_runs(base_dir: str, reach_runs_glob: str, other_env_runs_glob: str) -> List[str]:
    # get names of checkpoints in other env
    other_env_runs = glob.glob(os.path.join(base_dir, other_env_runs_glob))
    # get names of checkpoints in reach env
    reach_runs = glob.glob(os.path.join(base_dir, reach_runs_glob))
    other_env = other_env_runs[0].split('_prior_')[0].split('/')[-1]
    other_env_runs_names = [run.split('/')[-2].replace(other_env + '_', '') for run in other_env_runs]
    reach_runs_names = [run.split('/')[-2].replace('reach_', '') for run in reach_runs]
    valid_reach_runs = [run for run in reach_runs_names if run in other_env_runs_names]
    valid_reach_runs = [os.path.join(base_dir, 'reach', f'reach_{run}') for run in valid_reach_runs]
    return valid_reach_runs


def run_prior_only_evaluations():
    base_dir = '/home/rontechnion/masked_mimic_inversion'
    # cluster_base_dir = '/lustre/fsw/portfolios/nvr/users/ctessler/models/rons_2'
    cluster_base_dir = None
    # envs_names_list = ["inversion_steering", "inversion_direction_facing", "reach"]
    # envs_names_list = ["reach"]
    envs_names_list = [
        "inversion_steering",
        "inversion_direction_facing",
        "reach",
        "inversion_strike",
        "inversion_long_jump"
    ]
    gpu_ids = [0, 1, 2, 3]
    # gpu_ids = [1, 2, 3]
    termination = True
    perturbations = {
        "None": None,
        "gravity_z": -15,
        "complex_terrain": True,
        "friction": 0.4,
    }
    # perturbations = {"None": None}
    # record_video = True
    record_video = False
    output_dir = "eval_runs"
    num_envs = 4 * 1024
    # num_envs = 20
    games_per_env = 1
    # games_per_env = 5
    os.makedirs(output_dir, exist_ok=True)
    all_cmds = []
    for perturbation_name, perturbation_val in perturbations.items():
        for env_name in envs_names_list:
            for prior_only in [False, True]:
                if prior_only and env_name in ["inversion_strike", "inversion_long_jump"]:
                    continue
                if perturbation_name == "complex_terrain" and env_name in ["inversion_strike", "inversion_long_jump"]:
                    continue

                checkpoints_arg = get_checkpoints_arg(base_dir, env_name, perturbation_name, prior_only, record_video)

                project_name = "_".join(["FINAL_", env_name.replace('inversion_', '')])

                if perturbation_name != "None":
                    project_name += f"_{perturbation_name}"
                    use_perturbation = True
                else:
                    use_perturbation = False

                if cluster_base_dir is not None:
                    checkpoints_arg = checkpoints_arg.replace(base_dir, cluster_base_dir)

                cmd = [
                    "python", "phys_anim/eval_checkpoints.py",
                    f"{checkpoints_arg}",
                    f"+gpu_ids=[{','.join(map(str, gpu_ids))}]",
                    f"+num_envs={num_envs}",
                    f"+games_per_env={games_per_env}",
                    f"+prior_only={prior_only}",
                    f"+wandb.project={project_name}",
                    f"+use_perturbations={use_perturbation}",
                    f"+record_video={record_video}",
                    f"+termination={termination}",
                    f"+record_dir={record_dir}",
                ]
                if perturbation_name != "None":
                    cmd.append(f"+perturbations.{perturbation_name}={perturbation_val}")
                # subprocess.run(cmd, check=True)
                all_cmds.append(' '.join(cmd))
    with open(os.path.join(output_dir, f'{"record_" if record_video else ""}all_runs.sh'), 'w') as f:
        f.write('\n'.join(all_cmds))


if __name__ == '__main__':
    run_prior_only_evaluations()
