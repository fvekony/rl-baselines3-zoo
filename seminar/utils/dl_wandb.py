import os
import wandb
import argparse

api = wandb.Api(timeout=60)

base_path = "/home/fvekony/rl-baselines3-zoo/seminar/artifacts"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=str, nargs='+', default=None,
                        help='Only download runs from these groups (default: all groups)')
    args = parser.parse_args()

    runs = api.runs("fvekony-bergische-universit-t-wuppertal/dqn-seaquest")

    selected_groups = set(args.groups) if args.groups else None

    for run in runs:
        group = run.group
        if not group:
            continue
        if selected_groups and group not in selected_groups:
            continue

        print(f"Processing {run.id} (group: {group}, seed: {run.config.get('seed', 'unknown')})...")

        group_folder = os.path.join(base_path, group)
        os.makedirs(group_folder, exist_ok=True)

        seed = run.config.get('seed', 'unknown')
        run_folder_name = f"{run.id}_{seed}"
        run_folder = os.path.join(group_folder, run_folder_name)
        os.makedirs(run_folder, exist_ok=True)

        try:
            artifact = api.artifact(f'{run.entity}/{run.project}/run-{run.id}-history:latest')
            artifact.download(root=os.path.join(run_folder, "history"))
            print(f"Downloaded history artifact")
        except Exception as e:
            print(f"Failed to download history artifact: {e}")

        try:
            for file in run.files():
                file.download(root=run_folder, replace=True)
            print(f"Downloaded files")
        except Exception as e:
            print(f"Failed to download files: {e}")

if __name__ == "__main__":
    main()

