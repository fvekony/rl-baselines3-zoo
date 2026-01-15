import gymnasium as gym
import numpy as np
import time
import sys

#register atari environments
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

def get_ale(env):
    """Find ALE interface in wrapped environment."""
    #try direct access first
    if hasattr(env, 'ale'):
        return env.ale
    
    #unwrap through all wrappers
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
        if hasattr(unwrapped, 'ale'):
            return unwrapped.ale
    
    #try unwrapped attribute
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'ale'):
        return env.unwrapped.ale
    
    return None

def print_object_table(ram, prev_ram):
    """Print object positions in table format."""
    print("\n" + "="*120)
    print("FULL RAM (0-120)")
    print("="*120)
    
    #print header
    print(f"{'Addr':>4} | {'Value':>5} | {'Prev':>5} | {'Δ':>4} | {'Addr':>4} | {'Value':>5} | {'Prev':>5} | {'Δ':>4} | {'Addr':>4} | {'Value':>5} | {'Prev':>5} | {'Δ':>4}")
    print("-" * 120)
    
    #display in 3 columns: 0-9, 10-19, 20-29
    for addr in range(0, 10):
        for offset in [0, 10, 20]:
            addr_n = addr + offset
            if addr_n <= 120:
                old_val = prev_ram[addr_n] if prev_ram is not None else 0
                new_val = ram[addr_n]
                delta = new_val - old_val
                
                marker = " *" if delta != 0 else "  "
                print(f"[{addr_n:3d}]{marker} | {new_val:5d} | {old_val:5d} | {delta:+4d} |", end="")
                
                if offset < 20:
                    print(" ", end="")
        print()
    
    #30-39, 40-49, 50-59
    print()
    for addr in range(30, 40):
        for offset in [0, 10, 20]:
            addr_n = addr + offset
            if addr_n <= 120:
                old_val = prev_ram[addr_n] if prev_ram is not None else 0
                new_val = ram[addr_n]
                delta = new_val - old_val
                
                marker = " *" if delta != 0 else "  "
                print(f"[{addr_n:3d}]{marker} | {new_val:5d} | {old_val:5d} | {delta:+4d} |", end="")
                
                if offset < 20:
                    print(" ", end="")
        print()
        
    #60-69, 70-79, 80-89
    print()
    for addr in range(60, 70):
        for offset in [0, 10, 20]:
            addr_n = addr + offset
            if addr_n <= 120:
                old_val = prev_ram[addr_n] if prev_ram is not None else 0
                new_val = ram[addr_n]
                delta = new_val - old_val
                
                marker = " *" if delta != 0 else "  "
                print(f"[{addr_n:3d}]{marker} | {new_val:5d} | {old_val:5d} | {delta:+4d} |", end="")
                
                if offset < 20:
                    print(" ", end="")
        print()
    
    #90-99, 100-109, 110-120
    print()
    for addr in range(90, 100):
        for offset in [0, 10, 20]:
            addr_n = addr + offset
            if addr_n <= 120:
                old_val = prev_ram[addr_n] if prev_ram is not None else 0
                new_val = ram[addr_n]
                delta = new_val - old_val
                
                marker = " *" if delta != 0 else "  "
                print(f"[{addr_n:3d}]{marker} | {new_val:5d} | {old_val:5d} | {delta:+4d} |", end="")
                
                if offset < 20:
                    print(" ", end="")
        print()

def print_game_state(ram, prev_ram, action_name=""):
    """Print known game state values."""
    print("\n" + "="*70)
    print(f"Action: {action_name}")
    print("="*70)
    
    print("\nKNOWN GAME STATE:")
    print(f"  Lives/Subs   [59]: {ram[59]:3d}")
    print(f"  Divers Saved [61]: {ram[61]:3d}")
    print(f"  Diver Onboard[62]: {ram[62]:3d}")
    print(f"  Player X     [70]: {ram[70]:3d}", end="")
    if prev_ram is not None:
        print(f"  (Δ{ram[70] - prev_ram[70]:+4d})")
    else:
        print()
    print(f"  Player Y     [97]: {ram[97]:3d}", end="")
    if prev_ram is not None:
        print(f"  (Δ{ram[97] - prev_ram[97]:+4d})")
    else:
        print()
    print(f"  Oxygen      [102]: {ram[102]:3d}")
    print(f"  Death Timer [105]: {ram[105]:3d}")
    
    print("\nOBJECT TRACKING:")
    print(f"  Object X?    [30]: {ram[30]:3d}", end="")
    if prev_ram is not None:
        print(f"  (Δ{ram[30] - prev_ram[30]:+4d})")
    else:
        print()
    print(f"  Object Y?  [31-33]: {ram[31]:3d}, {ram[32]:3d}, {ram[33]:3d}")
    print(f"  Counters? [73-74]: {ram[73]:3d}, {ram[74]:3d}")
    print(f"  Anim?     [89-94]: {ram[89]:3d}, {ram[90]:3d}, {ram[91]:3d}, {ram[92]:3d}, {ram[93]:3d}, {ram[94]:3d}")

def print_ram_overlay(ram, prev_ram, action_name=""):
    """Print RAM values that changed."""
    print("\n" + "="*70)
    print(f"Action: {action_name}")
    print("="*70)
    
    #show all RAM addresses that changed
    changes = []
    for i in range(128):
        if prev_ram is None or ram[i] != prev_ram[i]:
            changes.append((i, prev_ram[i] if prev_ram is not None else 0, ram[i]))
    
    if changes:
        print("\nCHANGED RAM VALUES:")
        for addr, old_val, new_val in changes:
            print(f"  [{addr:3d}]: {old_val:3d} -> {new_val:3d}  (change: {new_val - old_val:+4d})")
    else:
        print("\nNo RAM changes detected")
    
    print("="*70)

#create environment with human rendering
env = gym.make("SeaquestNoFrameskip-v4", render_mode="human")
ale = get_ale(env)

if not ale:
    print("ERROR: Could not access ALE interface")
    sys.exit(1)

#seaquest action mappings
action_map = {
    'w': 2,   # UP
    's': 5,   # DOWN
    'a': 4,   # LEFT
    'd': 3,   # RIGHT
    'i': 10,  # UPFIRE
    'k': 13,  # DOWNFIRE
    'j': 12,  # LEFTFIRE
    'l': 11,  # RIGHTFIRE
    'f': 1,   # FIRE
    '.': 0,   # NOOP
}

print("\n" + "="*50)
print("SEAQUEST MANUAL PLAY - OBJECT TRACKER")
print("="*50)
print("\nControls (type letter then Enter):")
print("  w - UP")
print("  s - DOWN") 
print("  a - LEFT")
print("  d - RIGHT")
print("  i - UP+FIRE")
print("  k - DOWN+FIRE")
print("  j - LEFT+FIRE")
print("  l - RIGHT+FIRE")
print("  f - FIRE")
print("  . - NOOP (do nothing)")
print("  t - Toggle display mode (table/changes/both)")
print("  q - Quit")
print("\nDisplay modes:")
print("  'both'   - Show game state + object table")
print("  'table'  - Show object positions table only")
print("  'changes'- Show all RAM changes (classic mode)")
print("="*50 + "\n")

obs, info = env.reset()
env.render()
ram = ale.getRAM()
prev_ram = None

display_mode = 'both'  # 'both', 'table', 'changes'

print_game_state(ram, prev_ram, "RESET")
print_object_table(ram, prev_ram)
prev_ram = ram.copy()

running = True
step_count = 0

#terminal-based input
while running:
    user_input = input("\nAction: ").lower().strip()
    
    if user_input == 'q':
        break
    
    if user_input == 't':
        modes = ['both', 'table', 'changes']
        current_idx = modes.index(display_mode)
        display_mode = modes[(current_idx + 1) % len(modes)]
        print(f"\n>>> Display mode: {display_mode}\n")
        continue
    
    action = action_map.get(user_input, 0)
    
    #execute action for multiple frames to see visible movement
    total_reward = 0
    for _ in range(15):  # Hold action for 15 frames
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        step_count += 1
        
        if terminated or truncated:
            break
    
    ram = ale.getRAM()
    action_name = env.unwrapped.get_action_meanings()[action]
    
    if display_mode == 'both':
        print_game_state(ram, prev_ram, action_name)
        print_object_table(ram, prev_ram)
    elif display_mode == 'table':
        print_object_table(ram, prev_ram)
    elif display_mode == 'changes':
        print_ram_overlay(ram, prev_ram, action_name)
    
    prev_ram = ram.copy()
    
    if total_reward != 0:
        print(f"\n>>> REWARD: {total_reward}")
    
    if terminated or truncated:
        print("\n!!! EPISODE ENDED !!!\n")
        obs, info = env.reset()
        env.render()
        ram = ale.getRAM()
        if display_mode == 'both':
            print_game_state(ram, prev_ram, "RESET")
            print_object_table(ram, prev_ram)
        elif display_mode == 'table':
            print_object_table(ram, prev_ram)
        elif display_mode == 'changes':
            print_ram_overlay(ram, prev_ram, "RESET")
        prev_ram = ram.copy()

env.close()
print(f"\nTotal steps: {step_count}")
print("Game closed.")
