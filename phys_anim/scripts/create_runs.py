import os


def main():
    ##### PARAMETER CHANGES #####
    # DEBUG = True
    DEBUG = False

    # envs = ["direction_facing"]
    # envs = ["path_follower"]
    # envs = ["path_follower", "direction_facing", "steering"]
    envs = ["reach"]

    use_chens_prior = [True, False]
    use_text = [True, False]
    use_current_pose_obs = [True, False]
    use_bigger_model = [True, False]
    train_actor = [True, False]
    ##### PARAMETER CHANGES #####

    output_file_path = "runs.sh" if not DEBUG else "debug_runs.sh"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    base_run_command = "python phys_anim/train_agent.py +robot=smpl +backbone=isaacgym"
    extra_args = []
    max_epochs = 20 if DEBUG else 4000
    base_run_command += " seed=${seed}" if not DEBUG else ""
    base_run_command += " auto_load_latest=False" if DEBUG else ""
    if not DEBUG:
        extra_args += ["wandb.wandb_entity=phys_inversion wandb.wandb_project=chens_runs"]

    if DEBUG:
        opts = ["small_run", "wdb", ]
    else:
        opts = ["full_run", "wdb", "combined_callbacks"]

    # create runs for each combination of hyperparameters
    for env in envs:
        for prior_flag in use_chens_prior:
            for text_flag in use_text:
                for current_pose in use_current_pose_obs:
                    for bigger_model in use_bigger_model:
                        for train_actor_flag in train_actor:
                            current_opts = opts.copy()
                            current_extra_args = extra_args.copy()
                            current_run_command = ""
                            current_run_command += base_run_command
                            current_run_command += f' +exp=inversion/{env}'
                            current_experiment_name = f"{env}_prior_{prior_flag}_text_{text_flag}_current_pose_{current_pose}_bigger_{bigger_model}_train_actor_{train_actor_flag}"
                            # add with ++ all the flags for easier filtering
                            current_run_command += f" ++prior={prior_flag} ++text={text_flag} ++current_pose={current_pose} ++bigger={bigger_model}"
                            if train_actor_flag:
                                current_run_command += f" ++algo_type=MaskedMimic_Finetune"
                            else:
                                current_run_command += f" ++algo_type=MaskedMimic_Inversion"
                            if DEBUG:
                                current_experiment_name += '_DEBUG'
                                current_extra_args += [f"algo.config.max_epochs={max_epochs}"]
                            current_run_command += (
                                    f" experiment_name={current_experiment_name}" + "_${seed}" +
                                    f" ++clean_exp_name={current_experiment_name}"
                            )
                            current_extra_args += [f"env.config.use_chens_prior={prior_flag}"]
                            current_extra_args += [f"env.config.use_text={text_flag}"]
                            if current_pose:
                                current_opts += ["masked_mimic/inversion/current_pose_obs"]
                            if bigger_model:
                                current_extra_args += [
                                    "algo.config.models.extra_input_model_for_transformer.config.units=[512,512,512]"]
                            if train_actor_flag:
                                current_opts += ["masked_mimic/inversion/train_actor"]

                            opt_string = "+opt=[" + ','.join(current_opts) + "]"
                            extra_args_string = ' '.join(current_extra_args)
                            current_run_command = " ".join(
                                [current_run_command, opt_string, extra_args_string, ]
                            )

                            with open(output_file_path, "a") as f:
                                f.write(current_run_command + "\n")


if __name__ == "__main__":
    main()
