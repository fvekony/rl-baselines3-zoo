import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_run_history(run_folder):
    history_path = os.path.join(run_folder, "history")
    parquet_files = glob.glob(os.path.join(history_path, "*.parquet"))
    
    if not parquet_files:
        return None
    
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('_step')
    
    return df

def plot_player_position_heatmap(runs_data, output_path, global_step_range=None):
    all_x = []
    all_y = []
    
    for run_name, run_info in runs_data.items():
        df = run_info['df'].copy()
        
        #forward-fill global_step to handle NaN values
        if 'global_step' in df.columns:
            df['global_step'] = df['global_step'].ffill()
        
        #filter by global_step range if provided
        if global_step_range is not None and 'global_step' in df.columns:
            range_start, range_end = global_step_range
            df = df[(df['global_step'] >= range_start) & (df['global_step'] <= range_end)]
        
        if 'game/player_x' in df.columns and 'game/player_y' in df.columns:
            x_vals = df['game/player_x'].dropna()
            y_vals = df['game/player_y'].dropna()
            all_x.extend(x_vals.values)
            all_y.extend(y_vals.values)
    
    if len(all_x) == 0:
        print("No player position data found")
        return
    
    #create 2D histogram with coarser bins for better visibility
    #seaquest playable area: x: 21-134, y: 13-108 (y=13 is top, y=108 is bottom)
    #extend y to 0 to see if player ever goes above playable area
    bin_size = 5
    x_bins = np.arange(21, 135, bin_size)
    y_bins = np.arange(0, 109, bin_size)
    
    heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=[x_bins, y_bins])
    
    #transpose for correct orientation
    heatmap = heatmap.T
    
    #use log scale to show variation better (add 1 to avoid log(0))
    heatmap_log = np.log10(heatmap + 1)
    
    #set color scale to exclude zeros (bins with no visits)
    #zeros in heatmap become log10(1) = 0.0
    nonzero_values = heatmap_log[heatmap_log > 0]
    if len(nonzero_values) > 0:
        vmin = nonzero_values.min()
        vmax = heatmap_log.max()
    else:
        vmin = 0
        vmax = 1
    
    #plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap_log, cmap='hot', interpolation='nearest', aspect='auto',
                   extent=[x_bins[0], x_bins[-1], y_bins[-1], y_bins[0]], origin='upper',
                   vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('X Position (Left to Right)', fontsize=12)
    ax.set_ylabel('Y Position (Top to Bottom)', fontsize=12)
    
    if global_step_range:
        title = f'Player Position Heatmap (Steps {global_step_range[0]}-{global_step_range[1]}, Log Scale)'
    else:
        title = 'Player Position Heatmap (All Runs, Log Scale)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log10(Count + 1)', rotation=270, labelpad=20)
    
    #add stats
    mean_x = np.mean(all_x)
    mean_y = np.mean(all_y)
    ax.plot(mean_x, mean_y, 'b+', markersize=20, markeredgewidth=3, label=f'Mean: ({mean_x:.1f}, {mean_y:.1f})')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if global_step_range:
        filename = f'player_position_heatmap_{global_step_range[0]}_{global_step_range[1]}.png'
    else:
        filename = 'player_position_heatmap.png'
    
    plt.savefig(os.path.join(output_path, filename), dpi=150)
    plt.close()
    
    print(f"  Created player position heatmap")
    print(f"    Mean position: X={mean_x:.1f}, Y={mean_y:.1f}")
    print(f"    Total positions: {len(all_x):,}")

def plot_action_distribution_radar(runs_data, output_path, group_name, global_step_range=None, n_actions=18):
    #aggregate action counts across all runs
    action_counts = {i: [] for i in range(n_actions)}
    action_names = {}
    
    for run_name, run_info in runs_data.items():
        df = run_info['df'].copy()
        
        #forward-fill global_step to handle NaN values
        if 'global_step' in df.columns:
            df['global_step'] = df['global_step'].ffill()
        
        #filter by global_step range if provided
        if global_step_range is not None and 'global_step' in df.columns:
            range_start, range_end = global_step_range
            df = df[(df['global_step'] >= range_start) & (df['global_step'] <= range_end)]
        
        #find action count columns (exclude aggregate metrics like unique_count, entropy)
        action_cols = [col for col in df.columns 
                      if col.startswith('actions/') 
                      and col.endswith('_count')
                      and not col.endswith('unique_count')]
        
        for col in action_cols:
            #extract original action index
            #format: actions/{i}_{action_name}_count (exp_wrapper)
            #format: actions/{i}_orig{orig_idx}_{action_name}_count (reduced_actions_wrapper)
            parts = col.split('/')[-1].split('_')
            
            #check if first part is a number
            if not parts[0].isdigit():
                continue
            
            if len(parts) > 1 and 'orig' in parts[1]:
                #reduced_actions_wrapper format: extract orig_idx and action name
                orig_idx = int(parts[1].replace('orig', ''))
                action_name = '_'.join(parts[2:-1])  #everything between orig_idx and _count
            else:
                #exp_wrapper format: first number is the action index
                orig_idx = int(parts[0])
                action_name = '_'.join(parts[1:-1])  #everything between index and _count
            
            #store action name
            if orig_idx not in action_names and action_name:
                action_names[orig_idx] = action_name
            
            #sum all counts for this action across the run
            total_count = df[col].sum()
            if orig_idx < n_actions:
                action_counts[orig_idx].append(total_count)
    
    #calculate mean counts for each action
    mean_counts = []
    for i in range(n_actions):
        if action_counts[i]:
            mean_counts.append(np.mean(action_counts[i]))
        else:
            mean_counts.append(0)
    
    #normalize to percentages
    total = sum(mean_counts)
    if total > 0:
        mean_counts = [c / total * 100 for c in mean_counts]
    
    #define directional ordering for radar chart (exclude NOOP and FIRE)
    #polar plots: 0deg=right, 90deg=bottom (clockwise with theta_direction=-1)
    #ordering ensures opposite actions are 180deg apart
    directional_order = [
        3,   #RIGHT (0deg - right side)
        11,  #RIGHTFIRE (22.5deg)
        8,   #DOWNRIGHT (45deg)
        16,  #DOWNRIGHTFIRE (67.5deg)
        5,   #DOWN (90deg - bottom)
        13,  #DOWNFIRE (112.5deg)
        9,   #DOWNLEFT (135deg)
        17,  #DOWNLEFTFIRE (157.5deg)
        4,   #LEFT (180deg - left side, opposite of RIGHT)
        12,  #LEFTFIRE (202.5deg, opposite of RIGHTFIRE)
        7,   #UPLEFT (225deg, opposite of DOWNRIGHT)
        15,  #UPLEFTFIRE (247.5deg, opposite of DOWNRIGHTFIRE)
        2,   #UP (270deg - top, opposite of DOWN)
        10,  #UPFIRE (292.5deg, opposite of DOWNFIRE)
        6,   #UPRIGHT (315deg, opposite of DOWNLEFT)
        14,  #UPRIGHTFIRE (337.5deg, opposite of DOWNLEFTFIRE)
    ]
    
    noop_index = 0
    fire_index = 1
    
    #reorder directional actions
    ordered_counts = [mean_counts[i] if i < len(mean_counts) else 0 for i in directional_order]
    ordered_labels = [action_names.get(i, f'A{i}') for i in directional_order]
    
    #get NOOP and FIRE values
    noop_value = mean_counts[noop_index] if noop_index < len(mean_counts) else 0
    fire_value = mean_counts[fire_index] if fire_index < len(mean_counts) else 0
    
    #create radar chart with 16 directional actions
    n_directional = len(directional_order)
    angles = np.linspace(0, 2 * np.pi, n_directional, endpoint=False).tolist()
    ordered_counts_plot = ordered_counts + [ordered_counts[0]]
    angles_plot = angles + [angles[0]]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    #set clockwise direction and start at right (0deg)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(0)
    
    #plot directional actions
    ax.plot(angles_plot, ordered_counts_plot, 'o-', linewidth=2, color='blue', label='Directional Actions')
    ax.fill(angles_plot, ordered_counts_plot, alpha=0.25, color='blue')
    
    #plot NOOP and FIRE as circles centered at origin with radius = value
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    
    if noop_value > 0:
        noop_x = [noop_value] * len(circle_angles)
        ax.plot(circle_angles, noop_x, linewidth=2, color='red', label=f'NOOP ({noop_value:.1f}%)', linestyle='--')
    
    if fire_value > 0:
        fire_x = [fire_value] * len(circle_angles)
        ax.plot(circle_angles, fire_x, linewidth=2, color='orange', label=f'FIRE ({fire_value:.1f}%)', linestyle='--')
    
    ax.set_xticks(angles)
    ax.set_xticklabels(ordered_labels, size=9)
    ax.set_ylim(0, max(ordered_counts_plot + [noop_value, fire_value]) * 1.1)
    ax.set_ylabel('Usage (%)', fontsize=11)
    
    if global_step_range:
        title = f'Action Distribution - {group_name} (Steps {global_step_range[0]}-{global_step_range[1]})'
    else:
        title = f'Action Distribution - {group_name}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    
    if global_step_range:
        filename = f'action_distribution_radar_{global_step_range[0]}_{global_step_range[1]}.png'
    else:
        filename = 'action_distribution_radar.png'
    
    plt.savefig(os.path.join(output_path, filename), dpi=150)
    plt.close()
    
    print(f"  Created action distribution radar chart")
    print(f"    Total actions analyzed: {int(total):,}")
    print(f"    Most used action: A{np.argmax(mean_counts)} ({max(mean_counts):.1f}%)")


def plot_divers_distribution(runs_data, output_path, global_step_range=None):
    #count unique steps for each divers_onboard value
    diver_counts = {i: set() for i in range(7)}  #0-6 divers
    successful_surfacings = 0
    
    for run_name, run_info in runs_data.items():
        df = run_info['df'].copy()
        
        #forward-fill global_step to handle NaN values
        if 'global_step' in df.columns:
            df['global_step'] = df['global_step'].ffill()
        
        #filter by global_step range if provided
        if global_step_range is not None and 'global_step' in df.columns:
            range_start, range_end = global_step_range
            df = df[(df['global_step'] >= range_start) & (df['global_step'] <= range_end)]
        
        #count steps for each divers_onboard value
        if 'game/divers_onboard' in df.columns and 'global_step' in df.columns:
            for i in range(7):
                steps_with_i_divers = df[df['game/divers_onboard'] == i]['global_step'].dropna().unique()
                diver_counts[i].update(steps_with_i_divers)
        
        #count successful surfacings (6 divers at surface with reward)
        if 'game/divers_onboard' in df.columns and 'game/player_y' in df.columns and 'game/step_reward' in df.columns:
            #create boolean series for surfacing events and below surface states
            surfacing_event = (df['game/divers_onboard'] == 6) & (df['game/player_y'] == 13) & (df['game/step_reward'] != 0)
            below_surface = df['game/player_y'] > 13
            
            #create a state tracker: 1 when below surface, 0 when at/above surface
            #shift below_surface to detect transitions
            was_below = below_surface.shift(1, fill_value=True)
            
            #count surfacings: surfacing event that occurs after being below surface
            unique_surfacings = surfacing_event & was_below
            successful_surfacings += unique_surfacings.sum()
    
    #prepare data for plotting
    labels = [f'{i} Divers' for i in range(7)] + ['6 Divers\nSurfaced']
    counts = [len(diver_counts[i]) for i in range(7)] + [successful_surfacings]
    
    #create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['steelblue'] * 7 + ['green']
    bars = ax.bar(range(len(labels)), counts, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Divers Onboard', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    
    if global_step_range:
        title = f'Divers Distribution (Steps {global_step_range[0]}-{global_step_range[1]})'
    else:
        title = 'Divers Distribution (Unique Steps per Diver Count + Successful Surfacings)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    #add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if global_step_range:
        filename = f'divers_distribution_{global_step_range[0]}_{global_step_range[1]}.png'
    else:
        filename = 'divers_distribution.png'
    
    plt.savefig(os.path.join(output_path, filename), dpi=150)
    plt.close()
    
    print(f"  Created divers distribution plot")
    print(f"    Unique steps by diver count: {[len(diver_counts[i]) for i in range(7)]}")
    print(f"    Successful surfacings (6 divers): {successful_surfacings}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True, help="group name to plot")
    parser.add_argument("--artifacts-path", type=str, 
                        default="/home/fvekony/rl-baselines3-zoo/seminar/artifacts",
                        help="base path to artifacts folder")
    parser.add_argument("--plots-path", type=str,
                        default="/home/fvekony/rl-baselines3-zoo/seminar/plots",
                        help="base path to save plots")
    parser.add_argument("--heatmap", nargs='*', type=int, metavar=('START', 'END'),
                        help="create player position heatmap (optionally specify global_step range as two numbers)")
    parser.add_argument("--radar", nargs='*', type=int, metavar=('START', 'END'),
                        help="create action distribution radar chart (optionally specify global_step range as two numbers)")
    parser.add_argument("--divers", nargs='*', type=int, metavar=('START', 'END'),
                        help="create divers distribution plot (optionally specify global_step range as two numbers)")
    args = parser.parse_args()
    
    #parse heatmap range if provided
    heatmap_range = None
    if args.heatmap is not None:
        if len(args.heatmap) == 2:
            heatmap_range = (args.heatmap[0], args.heatmap[1])
            print(f"Heatmap will be filtered to global_step range: {heatmap_range[0]} to {heatmap_range[1]}")
        elif len(args.heatmap) != 0:
            print("Error: --heatmap requires either no arguments or exactly 2 arguments (range_start range_end)")
            return
    
    #parse radar range if provided
    radar_range = None
    if args.radar is not None:
        if len(args.radar) == 2:
            radar_range = (args.radar[0], args.radar[1])
            print(f"Radar will be filtered to global_step range: {radar_range[0]} to {radar_range[1]}")
        elif len(args.radar) != 0:
            print("Error: --radar requires either no arguments or exactly 2 arguments (range_start range_end)")
            return
    
    #parse divers range if provided
    divers_range = None
    if args.divers is not None:
        if len(args.divers) == 2:
            divers_range = (args.divers[0], args.divers[1])
            print(f"Divers will be filtered to global_step range: {divers_range[0]} to {divers_range[1]}")
        elif len(args.divers) != 0:
            print("Error: --divers requires either no arguments or exactly 2 arguments (range_start range_end)")
            return
    
    group_folder = os.path.join(args.artifacts_path, args.group)
    output_folder = os.path.join(args.plots_path, args.group)
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(group_folder):
        print(f"Group folder not found: {group_folder}")
        return
    
    #load all runs in the group
    runs_data = {}
    run_folders = [d for d in os.listdir(group_folder) 
                   if os.path.isdir(os.path.join(group_folder, d))]
    
    print(f"Loading {len(run_folders)} runs from group '{args.group}'...")
    
    for run_folder_name in run_folders:
        run_folder_path = os.path.join(group_folder, run_folder_name)
        df = load_run_history(run_folder_path)
        
        if df is not None:
            #extract run_id and seed from folder name (format: runid_seed)
            parts = run_folder_name.split('_')
            run_id = parts[0]
            seed = parts[1] if len(parts) > 1 else "unknown"
            
            runs_data[run_folder_name] = {
                'df': df,
                'run_id': run_id,
                'seed': seed,
                'folder': run_folder_path
            }
            print(f"  Loaded {run_folder_name}: {len(df)} rows")
        else:
            print(f"  No history found for {run_folder_name}")
    
    print(f"\nLoaded {len(runs_data)} runs successfully")
    
    if len(runs_data) == 0:
        print("No data to plot")
        return
    
    #create plots, args.group
    #
    print(f"\nCreating plots in {output_folder}...")
    
    if args.heatmap is not None:
        plot_player_position_heatmap(runs_data, output_folder, heatmap_range)
    
    if args.radar is not None:
        plot_action_distribution_radar(runs_data, output_folder, args.group, radar_range)
    
    if args.divers is not None:
        plot_divers_distribution(runs_data, output_folder, divers_range)
    
    #
    #TODO: add more custom plotting code here
    #
    #example structure:
    # for run_name, run_info in runs_data.items():
    #     df = run_info['df']
    #     seed = run_info['seed']
    #     plt.plot(df['_step'], df['your_metric'], label=f"seed {seed}")
    #
    # plt.legend()
    # plt.savefig(os.path.join(output_folder, "your_plot.png"))
    #
    
    print(f"\nAll plots saved to {output_folder}")

if __name__ == "__main__":
    main()
