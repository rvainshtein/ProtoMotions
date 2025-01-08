def main():
    output_file_path = 'runs.txt'
    base_run_command = 'python phys_anim/train_agent.py +robot=smpl +backbone=isaacgym +opt=[full_run,wdb]'
    experiment_arg = '+exp=inversion/{}'
    extra_args = '{}'

    envs = ['direction_facing']
    use_chens_prior = [True, False]
    use_text = [True, False]
    use_current_pose_obs = [True, False]
    use_bigger_model = [True, False]
    # create runs for each combination of hyperparameters
    for env in envs:
        for prior_flag in use_chens_prior:
            for text_flag in use_text:
                for current_pose in use_current_pose_obs:
                    for bigger_model in use_bigger_model:
                        current_run_command = ''
                        current_run_command += base_run_command
                        current_exp = experiment_arg.format(env)
                        experiment_name = f'{env}_prior_{prior_flag}_text_{text_flag}_current_pose_{current_pose}_bigger_{bigger_model}'
                        current_run_command += f' +experiment_name={experiment_name}'
                        if bigger_model:
                            current_run_command += ' algo.config.models.extra_input_model_for_transformer.config.units = [512, 512, 512]'
                        extra_args = f' +env.config.use_chens_prior={prior_flag}'
                        extra_args += f' +env.config.use_text={text_flag}'
                        if current_pose:
                            extra_args += f'+env.config.use_current_pose_obs={current_pose}'
                            extra_args += ' algo.config.models.extra_input_obs_size = 11'
                        if bigger_model:
                            extra_args += ' algo.config.models.extra_input_model_for_transformer.config.units = [512, 512, 512]'
                        current_run_command = ' '.join([current_run_command, current_exp, extra_args])
                        with open(output_file_path, 'a') as f:
                            f.write(current_run_command + '\n')


if __name__ == '__main__':
    main()
