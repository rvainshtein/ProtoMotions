import glob
import os
import subprocess

from phys_anim.scripts.exps.baselines import get_valid_reach_runs


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
            checkpoints_arg = f"checkpoint_paths=[{','.join(reach_runs)}]"
    return checkpoints_arg


def run_prior_only_evaluations():
    base_dir = '/home/rontechnion/masked_mimic_inversion'
    # clsuter_base_dir = '/lustre/fsw/portfolios/nvr/users/ctessler/models/rons_2'
    clsuter_base_dir = None
    envs_names_list = ["inversion_steering", "inversion_direction_facing", "reach"]
    # gpu_ids = [0, 1, 2, 3]
    gpu_ids = [0, ]
    # perturbations = {"None": None,
    #                  "gravity_z": -15,
    #                  "complex_terrain": True}
    perturbations = {"None": None}
    record_video = True
    output_dir = "eval_runs"
    num_envs = 20
    os.makedirs(output_dir, exist_ok=True)
    all_cmds = []
    for prior_only in [True, False]:
        for env_name in envs_names_list:
            for perturbation in perturbations:
                checkpoints_arg = get_checkpoints_arg(base_dir, env_name, perturbation, prior_only, record_video)

                project_name = "_".join(["final_eval", env_name.replace('inversion_', '')])

                if perturbation != "None":
                    perturbation_name = perturbation.split('=')[0]
                    project_name += f"_{perturbation_name}"
                    use_perturbation = True
                else:
                    use_perturbation = False

                if clsuter_base_dir is not None:
                    checkpoints_arg = checkpoints_arg.replace(base_dir, clsuter_base_dir)

                cmd = [
                    "python", "phys_anim/eval_checkpoints.py",
                    f"{checkpoints_arg}",
                    f"+gpu_ids=[{','.join(map(str, gpu_ids))}]",
                    f"+num_envs={num_envs}",
                    f"+prior_only={prior_only}",
                    f"+wandb.project={project_name}",
                    f"+use_perturbations={use_perturbation}",
                    f"+record_video={record_video}",
                ]
                if perturbation != "None":
                    cmd.append(f"+perturbations.{perturbation_name}={perturbation}")
                # subprocess.run(cmd, check=True)
                all_cmds.append(' '.join(cmd))
    with open(os.path.join(output_dir, f'{"record_" if record_video else ""}all_runs.sh'), 'w') as f:
        f.write('\n'.join(all_cmds))


if __name__ == '__main__':
    run_prior_only_evaluations()
