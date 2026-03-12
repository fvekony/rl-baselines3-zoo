import gymnasium as gym
import numpy as np
import json
import os
from datetime import datetime

#register atari environments
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

def get_ale(env):
    if hasattr(env, 'ale'):
        return env.ale
    
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
        if hasattr(unwrapped, 'ale'):
            return unwrapped.ale
    
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'ale'):
        return env.unwrapped.ale
    
    return None

#create environment
env = gym.make("SeaquestNoFrameskip-v4", render_mode="human")
ale = get_ale(env)

if not ale:
    print("ERROR: Could not access ALE interface")
    exit(1)

#log file
os.makedirs("seminar/logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"seminar/logs/annotated_ram_{timestamp}.jsonl"

print(f"\nLogging to: {log_file}")
print("\n=== INTERACTIVE RAM ANNOTATION ===")
print("\nControls:")
print("  w/a/s/d - Move")
print("  f - Fire")
print("  . - NOOP (do nothing)")
print("  n - Take snapshot and annotate what you see")
print("  q - Quit\n")
print("When you press 'n', describe what's on screen:")
print("  Example: '2 sharks left-to-right, 1 diver at bottom'")
print("="*60 + "\n")

action_map = {
    'w': 2, 's': 5, 'a': 4, 'd': 3,
    'f': 1, '.': 0,
}

obs, info = env.reset()
env.render()
step_count = 0
snapshots = []

running = True
while running:
    user_input = input("\nAction (or 'n' to annotate): ").lower().strip()
    
    if user_input == 'q':
        break
    
    if user_input == 'n':
        #take snapshot
        ram = ale.getRAM()
        
        print("\n" + "="*60)
        print("DESCRIBE WHAT YOU SEE ON SCREEN:")
        print("(enemies, divers, positions, directions, etc.)")
        description = input("> ")
        
        snapshot = {
            "step": step_count,
            "description": description,
            "ram": ram.tolist(),
            "enemies_30_39": [int(ram[i]) for i in range(30, 40)],
            "flags_40_49": [int(ram[i]) for i in range(40, 50)],
            "player_pos": [int(ram[70]), int(ram[97])],
            "score": [int(ram[56]), int(ram[57]), int(ram[58])],
            "lives": int(ram[59]),
            "oxygen": int(ram[102]),
        }
        
        snapshots.append(snapshot)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(snapshot) + "\n")
        
        print(f"\nSnapshot #{len(snapshots)} saved!")
        print(f"  Enemies 30-39: {snapshot['enemies_30_39']}")
        print(f"  Flags 40-49:   {snapshot['flags_40_49']}")
        print("="*60 + "\n")
        continue
    
    #empty input = NOOP
    if user_input == '':
        action = 0
    else:
        action = action_map.get(user_input, 0)
    
    #execute action
    for _ in range(15):
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step_count += 1
        
        if terminated or truncated:
            break
    
    if reward != 0:
        print(f">>> REWARD: {reward}")
    
    if terminated or truncated:
        print("\n!!! EPISODE ENDED !!!\n")
        obs, info = env.reset()
        env.render()

env.close()
print(f"\n\nSaved {len(snapshots)} annotated snapshots to {log_file}")
print("\nReview your annotations:")
for i, snap in enumerate(snapshots):
    print(f"\n{i+1}. {snap['description']}")
    print(f"   Enemies: {snap['enemies_30_39']}")
