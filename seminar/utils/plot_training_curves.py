import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn
import logging

logger = logging.getLogger(__name__)


def load_run_history(run_folder):
    history_path = os.path.join(run_folder, "history")
    parquet_files = glob.glob(os.path.join(history_path, "*.parquet"))
    
    if not parquet_files:
        return None
    
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('_step')
    
    return df


def load_group_data(group_folder):
    runs_data = {}
    run_folders = [d for d in os.listdir(group_folder) 
                   if os.path.isdir(os.path.join(group_folder, d))]
    
    for run_folder_name in run_folders:
        run_folder_path = os.path.join(group_folder, run_folder_name)
        df = load_run_history(run_folder_path)
        
        if df is not None:
            parts = run_folder_name.split('_')
            run_id = parts[0]
            seed = parts[1] if len(parts) > 1 else "unknown"
            
            runs_data[run_folder_name] = {
                'df': df,
                'run_id': run_id,
                'seed': seed,
                'folder': run_folder_path
            }
    
    return runs_data


def aggregate_metric(runs_data, metric_name, x_axis='global_step', max_steps=None, resample_freq=1000, use_std=False):
    #collect all data points from all runs
    all_data = []
    
    #debug: check available columns in first run
    first_run = list(runs_data.values())[0] if runs_data else None
    if first_run is not None:
        logger.debug(f"  Available columns: {list(first_run['df'].columns)[:20]}...")  #show first 20
        logger.debug(f"  Looking for metric: '{metric_name}', x-axis: '{x_axis}'")
    
    for run_name, run_info in runs_data.items():
        df = run_info['df']
        
        if metric_name not in df.columns:
            logger.debug(f"  Metric '{metric_name}' not in columns for run {run_name}")
            continue
            
        if x_axis not in df.columns:
            logger.debug(f"  X-axis '{x_axis}' not in columns for run {run_name}")
            continue
        
        #filter to rows where metric is not NaN
        data = df[[x_axis, metric_name]].copy()
        
        #forward-fill x-axis to handle cases where metric and x-axis are logged separately
        data[x_axis] = data[x_axis].ffill()
        
        #now filter to rows where metric is not NaN and x-axis has been filled
        data = data[(data[metric_name].notna()) & (data[x_axis].notna())]
        
        logger.debug(f"  Run {run_name}: {len(data)} rows with '{metric_name}' and forward-filled '{x_axis}'")
        
        if len(data) == 0:
            continue
        
        #handle duplicate x-axis values by averaging metric values
        data_grouped = data.groupby(x_axis, as_index=False)[metric_name].mean()
        
        if len(data_grouped) == 0:
            continue
        
        if max_steps is not None:
            data_grouped = data_grouped[data_grouped[x_axis] <= max_steps]
        
        if len(data_grouped) > 0:
            all_data.append(data_grouped)
    
    logger.debug(f"  Total runs with data: {len(all_data)}")
    
    if len(all_data) == 0:
        return None, None, None
    
    #find full x-axis range (use union instead of intersection)
    min_x = min([data[x_axis].min() for data in all_data])
    max_x = max([data[x_axis].max() for data in all_data])
    
    logger.debug(f"  X-axis range: {min_x} to {max_x}")
    
    if min_x >= max_x or np.isnan(min_x) or np.isnan(max_x):
        logger.warning(f"  Invalid x-axis range for metric '{metric_name}': min={min_x}, max={max_x}")
        return None, None, None
    
    #create common x-axis with resampling
    common_x = np.arange(min_x, max_x + resample_freq, resample_freq)
    
    logger.debug(f"  Common x-axis points: {len(common_x)}")
    
    #interpolate each run to common x-axis
    interpolated_runs = []
    for data in all_data:
        interp_y = np.interp(common_x, data[x_axis].values, data[metric_name].values)
        interpolated_runs.append(interp_y)
    
    logger.debug(f"  Interpolated {len(interpolated_runs)} runs")
    
    #calculate mean and std error
    interpolated_runs = np.array(interpolated_runs)
    mean_y = np.mean(interpolated_runs, axis=0)
    std_y = np.std(interpolated_runs, axis=0)
    std_error = std_y / np.sqrt(len(interpolated_runs))
    
    #return std or std_error based on parameter
    spread = std_y if use_std else std_error
    
    return common_x, mean_y, spread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=str, nargs='+', required=True, 
                        help="group names to plot")
    parser.add_argument("--labels", type=str, nargs='+',
                        help="labels for each group (defaults to group names)")
    parser.add_argument("--metric", type=str, nargs='+', default=["rollout/ep_rew_mean", "0"],
                        help="metrics to plot as pairs: metric_name smooth_value (e.g., eval/mean_reward 40)")
    parser.add_argument("--x-axis", type=str, default="global_step",
                        help="x-axis column name")
    parser.add_argument("--artifacts-path", type=str, 
                        default="/home/fvekony/rl-baselines3-zoo/seminar/artifacts",
                        help="base path to artifacts folder")
    parser.add_argument("--plots-path", type=str,
                        default="/home/fvekony/rl-baselines3-zoo/seminar/plots",
                        help="base path to save plots")
    parser.add_argument("--folder", type=str, default=None,
                        help="subfolder within plots-path to save plots")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="maximum number of steps to plot")
    parser.add_argument("--resample-freq", type=int, default=1000,
                        help="resampling frequency for x-axis alignment")
    parser.add_argument("--file-name", type=str, default=None,
                        help="output filename (defaults to metric name)")
    parser.add_argument("--title", type=str, default=None,
                        help="plot title")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="enable debug logging")
    parser.add_argument("--no-million", action="store_true", default=False,
                        help="do not convert x-axis to millions")
    parser.add_argument("--std", action="store_true", default=False,
                        help="plot standard deviation instead of standard error")
    args = parser.parse_args()
    
    #configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s'
    )
    
    if args.labels is None:
        args.labels = args.groups
    
    if len(args.labels) != len(args.groups):
        logger.error("Error: number of labels must match number of groups")
        return
    
    #parse metric pairs (metric_name, smooth_value)
    if len(args.metric) % 2 != 0:
        logger.error("Error: --metric requires pairs of (metric_name smooth_value)")
        return
    
    metric_configs = []
    for i in range(0, len(args.metric), 2):
        metric_name = args.metric[i]
        smooth_value = int(args.metric[i + 1])
        metric_configs.append((metric_name, smooth_value))
    
    seaborn.set()
    
    output_folder = args.plots_path
    if args.folder:
        output_folder = os.path.join(output_folder, args.folder)
    os.makedirs(output_folder, exist_ok=True)
    
    #loop over all metrics to plot
    for metric, smooth_window in metric_configs:
        logger.info(f"\nPlotting metric: {metric}")
        
        plt.figure(figsize=(12, 8))
        
        for group_name, label in zip(args.groups, args.labels):
            group_folder = os.path.join(args.artifacts_path, group_name)
            
            if not os.path.exists(group_folder):
                logger.error(f"Group folder not found: {group_folder}")
                continue
            
            logger.info(f"Loading group '{group_name}'...")
            runs_data = load_group_data(group_folder)
            logger.info(f"  Loaded {len(runs_data)} runs")
            
            if len(runs_data) == 0:
                continue
            
            x, mean_y, std_error = aggregate_metric(
                runs_data, metric, args.x_axis, args.max_steps, args.resample_freq, args.std
            )
            
            if x is None:
                logger.warning(f"  Metric '{metric}' not found in group '{group_name}'")
                continue
            
            #apply smoothing if requested
            if smooth_window and smooth_window > 1:
                window = smooth_window
                mean_y_smooth = pd.Series(mean_y).rolling(window=window, center=True, min_periods=1).mean().values
                std_error_smooth = pd.Series(std_error).rolling(window=window, center=True, min_periods=1).mean().values
            else:
                mean_y_smooth = mean_y
                std_error_smooth = std_error
            
            divider = 1.0 if args.no_million else 1e6
            
            plt.plot(x / divider, mean_y_smooth, label=label, linewidth=3)
            plt.fill_between(x / divider, mean_y_smooth + std_error_smooth, mean_y_smooth - std_error_smooth, alpha=0.5)
            
            logger.info(f"  Final value: {mean_y[-1]:.2f} +/- {std_error[-1]:.2f}")
        
        x_label_suffix = "" if args.no_million else "(in Million)"
        plt.xlabel(f"Timesteps {x_label_suffix}", fontsize=14)
        plt.ylabel(metric.replace('/', ' ').replace('_', ' ').title(), fontsize=14)
        
        if args.title:
            plt.title(args.title, fontsize=14)
        else:
            plt.title(f"Training Curves - {metric}", fontsize=14)
        
        plt.legend()
        plt.grid(True)
        
        #determine output filename
        if args.file_name:
            output_filename = args.file_name
            if args.std and not output_filename.endswith('_std.png'):
                output_filename = output_filename.replace('.png', '_std.png')
        else:
            suffix = '_std' if args.std else ''
            output_filename = metric.replace('/', '_') + suffix + '.png'
        
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_path}")
    
    logger.info(f"\nAll {len(metric_configs)} plots created successfully")


if __name__ == "__main__":
    main()
