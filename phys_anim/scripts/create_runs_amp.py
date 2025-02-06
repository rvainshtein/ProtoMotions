import os


def main():
    ##### PARAMETER CHANGES #####
    # DEBUG = True
    DEBUG = False

    # envs = ["direction_facing"]
    # envs = ["path_follower"]
    # envs = ["path_follower", "steering"]
    envs = ["reach"]
    use_disable_discriminator = [True, False]
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
        for disable_discriminator in use_disable_discriminator:
            current_opts = opts.copy()
            current_extra_args = extra_args.copy()
            current_run_command = ""
            current_run_command += base_run_command
            current_run_command += f' +exp=amp_inversion/{env}'
            current_experiment_name = f"{env}_disable_discriminator_{disable_discriminator}"

            if DEBUG:
                current_experiment_name += '_DEBUG'
                current_extra_args += [f"algo.config.max_epochs={max_epochs}"]
            current_run_command += (
                    f" experiment_name={current_experiment_name}" + "_${seed}"
            )
            if disable_discriminator:
                current_opts += ["disable_discriminator", "disable_discriminator_weights"]

            opt_string = "+opt=[" + ','.join(current_opts) + "]"
            extra_args_string = ' '.join(current_extra_args)
            current_run_command = " ".join(
                [current_run_command, opt_string, extra_args_string, ]
            )

            with open(output_file_path, "a") as f:
                f.write(current_run_command + "\n")


if __name__ == "__main__":
    main()
