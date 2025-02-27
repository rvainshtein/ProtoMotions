import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

def smooth_dataframe(df, x_col='frames', y_col='Episode Reward', window_size=1000000):
    """
    Apply a moving average smoothing to the dataframe based on x-values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to smooth
    x_col : str
        The column name for the x-axis values (default: 'frames')
    y_col : str
        The column name for the y-axis values to smooth (default: 'Episode Reward')
    window_size : int
        The window size in x-axis units for the moving average (default: 1000000 steps)
        
    Returns:
    --------
    pandas.DataFrame
        A new dataframe with smoothed values
    """
    # Create a copy to avoid modifying the original dataframe
    smoothed_df = df.copy()
    
    # Group by run_id and algo_str to smooth each run separately
    if 'run_id' in df.columns and 'algo_str' in df.columns:
        groups = df.groupby(['run_id', 'algo_str'])
    elif 'run_id' in df.columns:
        groups = df.groupby('run_id')
    elif 'algo_str' in df.columns:
        groups = df.groupby('algo_str')
    else:
        # If no grouping columns, treat the entire dataframe as one group
        groups = [(None, df)]
    
    result_dfs = []
    
    for name, group in tqdm(groups, desc="Smoothing runs", leave=False):
        # Sort by x_col to ensure proper smoothing
        group = group.sort_values(by=x_col).copy()
        
        # For each unique x value, calculate the smoothed value using vectorized operations
        unique_x = np.sort(group[x_col].unique())
        smoothed_values = {}
        
        for x_val in unique_x:
            # Find all points within the window
            window_mask = (group[x_col] >= x_val - window_size/2) & (group[x_col] <= x_val + window_size/2)
            if window_mask.any():
                smoothed_values[x_val] = group.loc[window_mask, y_col].mean()
            else:
                # If no points in window, use the original values
                smoothed_values[x_val] = group.loc[group[x_col] == x_val, y_col].iloc[0] if not group.loc[group[x_col] == x_val, y_col].empty else np.nan
        
        # Apply the smoothed values to the dataframe
        group[y_col] = group[x_col].map(smoothed_values)
        result_dfs.append(group)
    
    # Combine all smoothed groups back into a single dataframe
    if result_dfs:
        return pd.concat(result_dfs)
    else:
        return smoothed_df

def smooth_algo_rewards(df_plot, algo_str, x_col='frames', y_col='Episode Reward', window_size=1000000, 
                   filter_outliers=False, outlier_window=1000000, outlier_threshold=0.9):
    """
    Apply smoothing to a specific algorithm's data and compute mean and std.
    
    Parameters:
    -----------
    df_plot : pandas.DataFrame
        The dataframe containing all algorithms' data
    algo_str : str
        The algorithm string identifier to filter by
    x_col : str
        The column name for the x-axis values (default: 'frames')
    y_col : str
        The column name for the y-axis values to smooth (default: 'Episode Reward')
    window_size : int
        The window size in x-axis units for the moving average (default: 1000000 steps)
    filter_outliers : bool
        Whether to filter outliers using forward/backward comparison (default: False)
    outlier_window : int
        The window size for outlier detection in x-axis units (default: 1000000 steps)
    outlier_threshold : float
        The threshold below which a point is considered an outlier (default: 0.9)
        
    Returns:
    --------
    pandas.DataFrame
        A dataframe with smoothed mean and std values for the specified algorithm
    """
    # Filter data for the specific algorithm
    algo_df = df_plot[df_plot['algo_str'] == algo_str].copy()
    
    if algo_df.empty:
        return pd.DataFrame(columns=[x_col, 'mean', 'std', 'algo_str'])
    
    if filter_outliers:
        # Group by run_id if it exists, otherwise process all data together
        groupby_cols = ['run_id'] if 'run_id' in algo_df.columns else []
        
        filtered_dfs = []
        for name, group in algo_df.groupby(groupby_cols) if groupby_cols else [(None, algo_df)]:
            # Create a copy of the group to modify
            group = group.copy()
            
            # Sort by x_col to ensure proper comparison
            group = group.sort_values(by=x_col)
            x_values = group[x_col].values
            y_values = group[y_col].values
            
            # Create a mask for points to update
            points_to_update = []
            new_values = []

            last_best = y_values[0]
            for i in range(len(x_values)):
                current_x = x_values[i]
                current_y = y_values[i]
                
                last_best = max(last_best, current_y)
                if current_y < outlier_threshold * last_best:
                    new_value = last_best * outlier_threshold

                    points_to_update.append(i)
                    new_values.append(new_value)
            
            # Update all points at once
            if points_to_update:
                group.iloc[points_to_update, group.columns.get_loc(y_col)] = new_values
                print(f"Updated {len(points_to_update)} outliers in group {name}")
            
            filtered_dfs.append(group)
        
        # Combine all filtered groups
        algo_df = pd.concat(filtered_dfs) if filtered_dfs else algo_df
    
    # First compute std before smoothing
    result_df = algo_df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
    result_df['std'] = result_df['std'].fillna(0)
    
    # Then apply temporal smoothing to the mean values
    smoothed_means = smooth_dataframe(result_df, x_col, 'mean', window_size)
    result_df['mean'] = smoothed_means['mean']
    
    result_df['algo_str'] = algo_str
    return result_df

def smooth_all_algos(df_plot, order, x_col='frames', y_col='Episode Reward', window_size=1000000):
    """
    Apply smoothing to all algorithms in the order list and return a combined dataframe.
    
    Parameters:
    -----------
    df_plot : pandas.DataFrame
        The dataframe containing all algorithms' data
    order : list
        List of algorithm strings to process in order
    x_col : str
        The column name for the x-axis values (default: 'frames')
    y_col : str
        The column name for the y-axis values to smooth (default: 'Episode Reward')
    window_size : int
        The window size in x-axis units for the moving average (default: 1000000 steps)
        
    Returns:
    --------
    pandas.DataFrame
        A combined dataframe with smoothed mean and std values for all algorithms
    """
    result_dfs = []
    
    for algo_str in tqdm(order, desc="Processing algorithms"):
        smoothed_algo_df = smooth_algo_rewards(df_plot, algo_str, x_col, y_col, window_size)
        if not smoothed_algo_df.empty:
            result_dfs.append(smoothed_algo_df)
    
    if result_dfs:
        return pd.concat(result_dfs)
    else:
        return pd.DataFrame(columns=[x_col, 'mean', 'std', 'algo_str'])

def set_axis_range(fig, df_original, df_smoothed, x_col='frames'):
    """
    Set the x-axis range of the figure based on the original dataframe.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to update
    df_original : pandas.DataFrame
        The original dataframe before smoothing
    df_smoothed : pandas.DataFrame
        The smoothed dataframe
    x_col : str
        The column name for the x-axis values (default: 'frames')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The updated figure with proper x-axis range
    """
    # Get the min and max x values from the original dataframe
    x_min = df_original[x_col].min()
    x_max = df_original[x_col].max()
    
    # Update the figure's x-axis range
    fig.update_layout(
        xaxis=dict(
            range=[x_min, x_max],
            title=fig.layout.xaxis.title
        )
    )
    
    return fig

def create_smoothed_convergence_plot(df_plot, order, algo_pretty_names, color_discrete_map, 
                                    window_size=1000000, x_col='frames', y_col='Episode Reward'):
    """
    Create a smoothed convergence plot for all algorithms.
    
    Parameters:
    -----------
    df_plot : pandas.DataFrame
        The dataframe containing all algorithms' data
    order : list
        List of algorithm strings to process in order
    algo_pretty_names : dict
        Dictionary mapping algorithm strings to pretty names for the legend
    color_discrete_map : dict
        Dictionary mapping pretty names to colors
    window_size : int
        The window size in x-axis units for the moving average (default: 1000000 steps)
    x_col : str
        The column name for the x-axis values (default: 'frames')
    y_col : str
        The column name for the y-axis values to smooth (default: 'Episode Reward')
        
    Returns:
    --------
    tuple
        (fig, df_smoothed) - The figure and the smoothed dataframe
    """
    import plotly.graph_objects as go
    
    # Create a new figure
    fig = go.Figure()
    
    # Store the original x range
    x_min = df_plot[x_col].min()
    x_max = df_plot[x_col].max()
    
    # Process each algorithm
    all_smoothed_dfs = []
    
    for algo_str in reversed(order):
        # Get smoothed data for this algorithm
        smoothed_algo_df = smooth_algo_rewards(df_plot, algo_str, x_col, y_col, window_size)
        
        if not smoothed_algo_df.empty:
            all_smoothed_dfs.append(smoothed_algo_df)
            pretty_name = algo_pretty_names.get(algo_str, algo_str)
            color = color_discrete_map.get(pretty_name, 'black')
            
            # Add the upper bound of the std
            fig.add_trace(go.Scatter(
                x=smoothed_algo_df[x_col],
                y=smoothed_algo_df['mean'] + smoothed_algo_df['std'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
            ))
            
            # Add the lower bound of the std
            color_alpha = color.replace(')', ', 0.2)').replace('rgb', 'rgba')
            fig.add_trace(go.Scatter(
                x=smoothed_algo_df[x_col],
                y=smoothed_algo_df['mean'] - smoothed_algo_df['std'],
                fillcolor=color_alpha,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            # Add the mean line
            fig.add_trace(go.Scatter(
                x=smoothed_algo_df[x_col],
                y=smoothed_algo_df['mean'],
                mode='lines',
                name=pretty_name,
                line=dict(width=2, color=color)
            ))
    
    # Combine all smoothed dataframes
    df_smoothed = pd.concat(all_smoothed_dfs) if all_smoothed_dfs else pd.DataFrame()
    
    # Set the x-axis range to match the original data
    fig.update_layout(
        xaxis=dict(
            range=[x_min, x_max],
            title='Step'
        ),
        yaxis=dict(
            title='Mean Reward'
        ),
        autosize=False,
        width=1200,
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            itemsizing='constant',
        ),
        legend_traceorder='reversed',
    )
    
    return fig, df_smoothed 