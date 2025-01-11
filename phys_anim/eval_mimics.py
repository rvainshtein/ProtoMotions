import subprocess

import typer
import yaml

from phys_anim.utils.file_utils import load_motions

models = {
    # "fabric_fbt_new_fixes_flat": {
    #     "get_data_commands": [],
    #     "model_path": "results/after_refactor/full_body_tracker/last.ckpt",
    #     "extra_args": "+scene_lib=null +force_flat_terrain=True",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["reduced_full_body"],
    #     "robot": "smpl",
    # },
    # "fabric_fbt_new_fixes_terrain": {
    #     "get_data_commands": [],
    #     "model_path": "results/after_refactor/full_body_tracker/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]'",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["reduced_full_body"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_full_body_flat": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+force_flat_terrain=True +scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=True +env.config.masked_mimic_masking.start_without_history_prob=1.0 +vae_latent_from_prior=False ",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["reduced_full_body"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_full_body_terrain": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]' +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=True +env.config.masked_mimic_masking.start_without_history_prob=1.0 +vae_latent_from_prior=False ",
    #     "max_motions_per_simulation": 2048,  # 1024,
    #     "supported_tracking_forms": ["reduced_full_body"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_vr_flat_full_body_metrics": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=True +env.config.masked_mimic_masking.start_without_history_prob=1.0 ",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["occulus"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_vr_terrain_full_body_metrics": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]'  +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=True +env.config.masked_mimic_masking.start_without_history_prob=1.0",
    #     "max_motions_per_simulation": 2048,  # 1024,
    #     "supported_tracking_forms": ["occulus"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_vr_flat": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0  +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["occulus"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_vr_terrain": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]'  +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0  +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 2048,  # 1024,
    #     "supported_tracking_forms": ["occulus"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_hands_flat": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0  +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["hands"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_hands_terrain": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]'  +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0  +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 2048,  # 1024,
    #     "supported_tracking_forms": ["hands"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_feet_flat": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0  +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["feet"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_feet_terrain": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]'  +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0  +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 2048,  # 1024,
    #     "supported_tracking_forms": ["feet"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_head_flat": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0  +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["head"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_head_terrain": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]'  +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0 +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 2048,  # 1024,
    #     "supported_tracking_forms": ["head"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_pelvis_flat": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0 +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 4096,  # 1024,
    #     "supported_tracking_forms": ["pelvis"],
    #     "robot": "smpl",
    # },
    # "fabric_maskedmimic_pelvis_terrain": {
    #     "get_data_commands": [],
    #     "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
    #     "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]'  +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0 +env.config.masked_mimic_obs.masked_mimic_report_full_body_metrics=False",
    #     "max_motions_per_simulation": 2048,  # 1024,
    #     "supported_tracking_forms": ["pelvis"],
    #     "robot": "smpl",
    # },
    "fabric_maskedmimic_objects_sit": {
        "get_data_commands": [],
        "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
        "extra_args": "+scene_lib.max_num_objects=30 +terrain.config.spacing_between_scenes=25 +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0 +force_flat_terrain=True",
        "eval_opts": "masked_mimic/tasks/object",
        "max_motions_per_simulation": 1024,  # 1024,
        "supported_tracking_forms": ["no_constraint"],
        "fixed_motion_id": 1,
        "motion_file": "/home/ctessler/Downloads/SAMP-filtered_new.pt",
        "robot": "smpl",
    },
    "fabric_maskedmimic_pacer_flat": {
        "get_data_commands": [],
        "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
        "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0 ",
        "eval_opts": "masked_mimic/tasks/path_follower",
        "max_motions_per_simulation": 4096,  # 1024,
        "supported_tracking_forms": ["pelvis"],
        "fixed_motion_id": 2768,
        "motion_file": "../Downloads/AMASS-train-occlusion_filtered.pt",
        "robot": "smpl",
    },
    "fabric_maskedmimic_pacer_terrain": {
        "get_data_commands": [],
        "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
        "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]' +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0",
        "eval_opts": "masked_mimic/tasks/path_follower",
        "max_motions_per_simulation": 4096,  # 1024,
        "supported_tracking_forms": ["pelvis"],
        "fixed_motion_id": 2768,
        "motion_file": "../Downloads/AMASS-train-occlusion_filtered.pt",
        "robot": "smpl",
    },
    "fabric_maskedmimic_reach": {
        "get_data_commands": [],
        "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
        "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0   +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0 ",
        "eval_opts": "masked_mimic/tasks/reach",
        "max_motions_per_simulation": 4096,  # 1024,
        "supported_tracking_forms": ["no_constraint"],
        "fixed_motion_id": 2768,
        "motion_file": "../Downloads/AMASS-train-occlusion_filtered.pt",
        "robot": "smpl",
    },
    "fabric_maskedmimic_reach_terrain": {
        "get_data_commands": [],
        "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
        "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]' +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0",
        "eval_opts": "masked_mimic/tasks/reach",
        "max_motions_per_simulation": 4096,  # 1024,
        "supported_tracking_forms": ["no_constraint"],
        "fixed_motion_id": 2768,
        "motion_file": "../Downloads/AMASS-train-occlusion_filtered.pt",
        "robot": "smpl",
    },
    "fabric_maskedmimic_steering": {
        "get_data_commands": [],
        "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
        "extra_args": "+scene_lib=null +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0  +force_flat_terrain=True +env.config.masked_mimic_masking.start_without_history_prob=1.0 ",
        "eval_opts": "masked_mimic/tasks/steering",
        "max_motions_per_simulation": 4096,  # 1024,
        "supported_tracking_forms": ["no_constraint"],
        "fixed_motion_id": 2768,
        "motion_file": "../Downloads/AMASS-train-occlusion_filtered.pt",
        "robot": "smpl",
    },
    "fabric_maskedmimic_steering_terrain": {
        "get_data_commands": [],
        "model_path": "results/research/masked_mimic_pointcloud/last.ckpt",
        "extra_args": "+scene_lib=null +terrain.config.terrain_proportions='[0.3,0.15,0.25,0.25,0.07,0.,0.,0.]' +env.config.masked_mimic_masking.motion_text_embeddings_visible_prob=0.0 +env.config.masked_mimic_masking.target_pose_visible_prob=0.0 +env.config.masked_mimic_masking.start_without_history_prob=1.0",
        "eval_opts": "masked_mimic/tasks/steering",
        "max_motions_per_simulation": 4096,  # 1024,
        "supported_tracking_forms": ["no_constraint"],
        "fixed_motion_id": 2768,
        "motion_file": "../Downloads/AMASS-train-occlusion_filtered.pt",
        "robot": "smpl",
    },
}
motion_files = [
    {
        "motion_file": "../Downloads/AMASS-train-occlusion_filtered.pt",
        "yaml_file": "data/yaml_files/amass_train.yaml",
    },
    {
        "motion_file": "../Downloads/AMASS-test-occlusion_filtered.pt",
        "yaml_file": "data/yaml_files/amass_test.yaml",
    },
]


def main(
    save_path: str = ".",
    experiment_name: str = None,
    tracking_form: str = None,
    iteration_to_run: int = None,
):
    import os

    # Ensure save_path directory exists
    if save_path != ".":
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.getcwd()

    print(f"Results will be saved in: {save_path}")

    if tracking_form is not None:
        tracking_forms = [tracking_form]
    else:
        tracking_forms = [
            "full_body",
            "head",
            "pelvis",
            "pelvis_position",
            "hands",
            "feet",
            "occulus",
            "reduced_full_body",
            "occulus_pelvis",
            "no_constraint",
        ]

    for model_name, model_params in models.items():
        existing_results = {}
        if experiment_name is not None and model_name != experiment_name:
            continue

        for get_data_command in model_params["get_data_commands"]:
            print(get_data_command)
            subprocess.call(get_data_command, shell=True)

        max_motions_per_simulation = model_params["max_motions_per_simulation"]

        for tracking_form in tracking_forms:
            if (
                "supported_tracking_forms" in model_params
                and tracking_form not in model_params["supported_tracking_forms"]
            ):
                continue

            for motion_file_data in motion_files:
                if "motion_file" in model_params:
                    motion_file = model_params["motion_file"]
                    max_motions_per_simulation = 1024
                    num_motions = max_motions_per_simulation
                else:
                    motion_file = motion_file_data["motion_file"]
                    yaml_file = motion_file_data["yaml_file"]

                    num_motions = len(load_motions(yaml_file))
                recorded_so_far = 0
                current_iteration = 0

                results = {}

                while recorded_so_far < num_motions:
                    if (
                        iteration_to_run is None
                        or iteration_to_run == current_iteration
                    ):
                        eval_opts = f"masked_mimic/constraints/{tracking_form}"
                        if "eval_opts" in model_params:
                            eval_opts = f"[{model_params['eval_opts']},{eval_opts}]"
                        command_line = (
                            f"python phys_anim/eval_tracker.py "
                            f"+robot={model_params['robot']} +backbone=isaacgym "
                            f"+opt={eval_opts} "
                            f"+motion_file={motion_file} "
                            f"+ngpu=1 "
                            f"{model_params['extra_args']} "
                            f"+headless=True "
                            f"+checkpoint={model_params['model_path']} "
                            f"+num_envs={min(max_motions_per_simulation, num_motions - recorded_so_far)} "
                            f"+env.config.max_episode_length={10000 if 'max_episode_length' not in model_params else model_params['max_episode_length']} "
                            "+algo.config.eval_num_episodes=5 "
                            f"{'' if 'eval_length' not in model_params else '+algo.config.eval_length='+str(model_params['eval_length'])} "
                        )
                        if "fixed_motion_id" not in model_params:
                            command_line += (
                                f"+env.config.motion_index_offset={recorded_so_far} "
                            )
                        else:
                            command_line += f"+env.config.fixed_motion_id={model_params['fixed_motion_id']} "

                        log_path = f"{save_path}/mimic_eval_results.log"

                        print(command_line)
                        subprocess.call(command_line + f" > {log_path}", shell=True)

                        results_print_started = False
                        for line in open(f"{log_path}"):
                            if results_print_started:
                                try:
                                    metric_name = None
                                    result_parts = line.split(":")
                                    if len(result_parts) == 2:
                                        metric_name = result_parts[0].strip()
                                        metric_value = float(result_parts[1].strip())
                                        if metric_name not in results:
                                            results[metric_name] = []

                                        results[metric_name].append(
                                            (
                                                min(
                                                    max_motions_per_simulation,
                                                    num_motions - recorded_so_far,
                                                ),
                                                metric_value,
                                            )
                                        )
                                except (ValueError, TypeError):
                                    if (
                                        metric_name is not None
                                        and metric_name in results
                                    ):
                                        del results[metric_name]
                            if "--- EVAL MIMIC RESULTS ---" in line:
                                results_print_started = True

                    recorded_so_far += max_motions_per_simulation
                    current_iteration += 1

                # merge both dicts, with the new one overwriting the old one
                if model_name not in existing_results:
                    existing_results[model_name] = {}
                if tracking_form not in existing_results[model_name]:
                    existing_results[model_name][tracking_form] = {}
                if motion_file not in existing_results[model_name][tracking_form]:
                    existing_results[model_name][tracking_form][motion_file] = {}

                for metric_name, metric_results in results.items():
                    # store the average score weighted by the number of motions
                    metric_result = [
                        sum([m[0] * m[1] for m in metric_results])
                        / sum([m[0] for m in metric_results])
                    ]
                    existing_results[model_name][tracking_form][motion_file][
                        metric_name
                    ] = metric_result

                if experiment_name is not None:
                    path = f"{save_path}/mimic_eval_results_{model_name}_{experiment_name}_{tracking_form}.yaml"
                else:
                    path = f"{save_path}/mimic_eval_results_{model_name}_{tracking_form}.yaml"
                with open(path, "w") as file:
                    yaml.dump(existing_results, file)

                if "motion_file" in model_params:
                    break


if __name__ == "__main__":
    typer.run(main)
