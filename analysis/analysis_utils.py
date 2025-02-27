import os
from glob import glob
from typing import List

import numpy as np
import pandas as pd
import wandb
from plotly import graph_objects as go
from tqdm.notebook import tqdm


def fetch_projects(entity, project_prefix):
    api = wandb.Api()
    projects = api.projects(entity=entity)
    projects = [project for project in projects if project.name.startswith(project_prefix)]
    return projects


def get_runs(entity, project, filters=None, keys=None, samples=None):
    if filters is None:
        filters = {"state": {"$eq": "finished"}}
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)
    data = []
    for run in tqdm(runs, desc=f"Fetching runs from {project}", position=1):
        history = run.history(keys=keys, pandas=True, samples=samples)  # Adjust as needed

        # Fetch hyperparameters
        config = run.config
        history = pd.concat([history, pd.json_normalize(config, sep='.')], axis=1)

        history["run_name"] = run.name
        history["run_id"] = run.id

        data.append(history)
    return pd.concat(data).reset_index() if data else pd.DataFrame()


def generate_raw_csvs(project_prefix='FINAL__', entity='phys_inversion', filters=None):
    if filters is None:
        filters = {"state": {"$eq": "finished"}}
    projects = fetch_projects(entity, project_prefix)
    os.makedirs(f"{project_prefix}/raw", exist_ok=True)
    for project in tqdm(projects, desc="Generating raw CSVs", position=0):
        df = get_runs(entity, project.name, filters)
        df.to_csv(f"{project_prefix}/raw/{project.name}.csv", index=False)


def generate_grouped_csvs(project_prefix='FINALLY__', perturbation_types=(), metrics_rename_dict=None):
    if metrics_rename_dict is None:
        metrics_rename_dict = {}
    raw_csvs_paths = os.listdir(f"{project_prefix}/raw")
    grouped_dir = f"{project_prefix}/grouped"
    os.makedirs(grouped_dir, exist_ok=True)
    if raw_csvs_paths is None:
        raw_csvs_paths = []
    for project_file in raw_csvs_paths:
        df = pd.read_csv(f"{project_prefix}/raw/{project_file}")
        # drop current reach_success column since only PULSE has it
        if 'reach_success' in df.columns:
            df = df.drop(columns=['reach_success'])
        # rename columns
        df.rename(columns=metrics_rename_dict, inplace=True)
        df[df['reach_success'] == "NaN"] = 0

        # multiply by 100 to get percentage
        df['reach_success'] = df['reach_success'] * 100

        # check if algo type is null, if so then if env.config.disable_discriminator is True then it's PPO else it's AMP
        df['algo_type'] = df.apply(lambda x:
                                   x['algo_type'] if pd.notna(x['algo_type'])
                                   else ('PPO' if x['env.config.disable_discriminator'] else 'AMP'), axis=1)
        df['prior'] = df['prior'].fillna(False)

        df = df[df.any(axis=1)]

        perturbation_config_keys = [f'env.config.perturbations.{perturb_type}' for perturb_type in perturbation_types]
        df_grouped = df.groupby(["algo_type", "prior", "use_perturbations", *perturbation_config_keys],
                                dropna=False).agg(
            {"reach_success": ["mean", "std"]})
        df_grouped.columns = ["_".join(col).strip() for col in df_grouped.columns.to_flat_index()]
        df_grouped.reset_index(inplace=True)

        os.makedirs(project_prefix, exist_ok=True)
        df_grouped.to_csv(f"{grouped_dir}/{project_file}", float_format="%.4f")


def generate_intermediate_mean_std_df(file, keep_cols, renamed_keep_cols):
    df = pd.read_csv(file)
    df = df.dropna(axis=1, how='all')  # Drop empty columns
    df = df[keep_cols]
    df.columns = renamed_keep_cols
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')  # Convert to numeric, coerce errors to NaN
    df['std'] = pd.to_numeric(df['std'], errors='coerce')  # Convert to numeric, coerce errors to NaN
    df['value'] = df.apply(
        lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}" if pd.notna(row['std']) else f"{row['mean']:.2f}",
        axis=1)
    df['algo_str'] = df['algo_type'] + '_prior_' + df['prior'].map(str)
    return df


def generate_combined_df(grouped_dir, project_prefix='FINALLY__'):
    # Read all CSV files
    csv_files = glob(os.path.join(grouped_dir, "*.csv"))
    keep_cols = ['algo_type', 'prior', 'use_perturbations', 'reach_success_mean', 'reach_success_std']
    renamed_keep_cols = ['algo_type', 'prior', 'use_perturbations', 'mean', 'std']
    # Dictionary to store data
    data = {}
    for file in csv_files:
        env_name = os.path.splitext(os.path.basename(file))[0].replace(project_prefix, '')  # Remove project prefix
        df = generate_intermediate_mean_std_df(file, keep_cols, renamed_keep_cols)
        df['env_perturb'] = env_name + '_perturb_' + df['use_perturbations'].map(str)

        for ep in df['env_perturb'].unique():
            data[ep] = df[df['env_perturb'] == ep].set_index(['algo_str', 'algo_type', 'prior'])['value']
    # Combine data into a single DataFrame
    combined_df = pd.concat(data, axis=1).reset_index()
    return combined_df


def reorder_rows(df, order):
    # Set order of rows
    df = df.sort_values('algo_str', key=lambda x: x.map(order.index))

    return df


def merge_rows(combined_df):
    # Merge 'PureRL_prior_False' and 'PPO_prior_False' rows
    df = combined_df.copy()
    if 'PureRL' in df['algo_type'].values and 'PPO' in df['algo_type'].values:
        pure_rl_row = df[df['algo_type'] == 'PureRL'].iloc[0]
        ppo_row = df[df['algo_type'] == 'PPO'].iloc[0]
        merged_row = ppo_row.combine_first(pure_rl_row)
        df = df[~df['algo_type'].isin(['PureRL', 'PPO'])]
        df = pd.concat([df, merged_row.to_frame().T], ignore_index=True)
    return df


def post_process_df(df, order):
    df = reorder_rows(df, order)
    df = merge_rows(df)
    return df


def parse_mean_std(value):
    try:
        mean, std = value.split('±')
        return float(mean), float(std)
    except ValueError:
        return np.nan, np.nan


def max_values_mask(df, ignore_cols=None):
    if ignore_cols is None:
        ignore_cols = []

    # Create a false mask like df
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        if col in ignore_cols:
            continue

        parsed = df[col].apply(parse_mean_std)
        means = parsed.apply(lambda x: x[0])
        stds = parsed.apply(lambda x: x[1])

        max_val = means.max()
        std_of_max = stds[means.idxmax()]
        mask[col] = (means >= max_val - std_of_max) & (means <= max_val + std_of_max)

    return mask


def generate_latex(df, output_file_path, caption=None, label=None, bold_mask=None, **kwargs):
    """
    Generate a LaTeX table from a DataFrame with optional bold values.

    Parameters:
    - df: DataFrame
    - output_file_path: str, path to save the LaTeX file
    - caption: str, optional caption for the table
    - label: str, optional label for the table
    - bold_mask: DataFrame-like, optional, same shape as df with True for values to be bold
    - **kwargs: additional arguments passed to DataFrame.to_latex
    """
    if bold_mask is not None:
        df = df.mask(bold_mask, df.applymap(lambda x: f"\\textbf{{{x}}}"))

    # Auto-generate column format: 'l' for the first column, 'c' for the rest
    column_format = 'l' + 'c' * (len(df.columns) - 1)
    df.to_latex(
        output_file_path,
        index=False,
        float_format="%.2f",
        column_format=column_format,
        caption=caption,
        label=label,
        escape=False,
        **kwargs
    )


def generate_table(final_df: pd.DataFrame, algos: List[str], envs: List[str], algo_rename_dict: dict,
                   env_rename_dict: dict):
    # different methods - PPO, AMP, PULSE, PRIOR_ONLY, INVERSION prior FALSE
    # 5 envs, no perturbations
    table_df = final_df.copy()
    table_df = table_df[table_df['algo_str'].isin(algos)]
    non_perturb_cols = [col for col in table_df.columns if 'perturb_False' in col]
    table_df = table_df[['algo_str'] + non_perturb_cols]

    renamed_cols = [col.replace('_perturb_False', '') for col in non_perturb_cols]
    # Rename columns
    table_df.columns = ['Method'] + renamed_cols
    # Rename algo names
    table_df['Method'] = table_df['Method'].replace(algo_rename_dict)
    table_df = table_df[['Method'] + envs]
    table_df = table_df.rename(columns=env_rename_dict)
    # table_df = table_df.set_index('Method')
    table_df = table_df.fillna('-')
    return table_df


def generate_combined_perturb_df(grouped_dir, perturbation_types=('gravity_z', 'friction'),
                                 perturbation_defaults=(-9.81, 1.0), project_prefix='PERTURB__'):
    # Read all CSV files
    csv_files = glob(os.path.join(grouped_dir, "*.csv"))
    perturbation_config_keys = [f'env.config.perturbations.{perturb_type}' for perturb_type in perturbation_types]
    keep_cols = ['algo_type', 'prior', *perturbation_config_keys, 'reach_success_mean', 'reach_success_std']
    renamed_keep_cols = ['algo_type', 'prior', *perturbation_config_keys, 'mean', 'std']
    # Dictionary to store data
    data = {}
    for file in csv_files:
        # TODO: handle multiple envs
        env_name = os.path.splitext(os.path.basename(file))[0].replace(project_prefix, '')  # Remove project prefix
        df = generate_intermediate_mean_std_df(file, keep_cols, renamed_keep_cols)
        df['mean'] = df['mean'].fillna(0)
        # aggregate perturbation types and values for each algo
    #     for perturb_key, perturb_name in zip(perturbation_config_keys, perturbation_types):
    #         df['perturb_name'] = env_name + f'_{perturb_name}_' + df[perturb_key].map(str)
    #
    #     for ep in df['env_perturb'].unique():
    #         data[ep] = df[df['env_perturb'] == ep].set_index(['algo_str', 'algo_type', 'prior'])['value']
    #
    # # Combine data into a single DataFrame
    # combined_df = pd.concat(data, axis=1).reset_index()
    # return combined_df
    return df


def create_beam_plot(df, x_col, y_col, std_col, title, x_label, y_label):
    fig = go.Figure()

    for algo in df['algo_str'].unique():
        algo_df = df[df['algo_str'] == algo]

        # Add the mean line
        fig.add_trace(go.Scatter(
            x=algo_df[x_col],
            y=algo_df[y_col],
            mode='lines',
            name=algo,
            line=dict(width=2)
        ))

        # Add the upper bound of the std
        fig.add_trace(go.Scatter(
            x=algo_df[x_col],
            y=algo_df[y_col] + algo_df[std_col],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))

        # Add the lower bound of the std
        fig.add_trace(go.Scatter(
            x=algo_df[x_col],
            y=algo_df[y_col] - algo_df[std_col],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        autosize=False,
        width=1200,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(title='% Change from Baseline Gravity', tickvals=[0.5, 1, 2]),
        yaxis=dict(gridcolor='black'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )

    fig.show()
