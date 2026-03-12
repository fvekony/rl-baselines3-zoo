# Seaquest RAM Address Map

Estimated reference for Atari 2600 Seaquest RAM addresses, discovered through manual testing and annotation.

---

## Core Game State

### Score System
- **RAM[56-58]**: Score in BCD (Binary Coded Decimal) format
  - 6 digits total: `RAM[56]` = ten-thousands + hundred-thousands, `RAM[57]` = thousands + hundreds, `RAM[58]` = tens + ones
  - **Decoding**: Convert to hex and read as decimal
    ```python
    score = int(f"{ram[56]:02x}{ram[57]:02x}{ram[58]:02x}")
    # Example: RAM[56]=0, RAM[57]=1, RAM[58]=0 → "000100" = 100 points
    ```
  - **Example values**:
    - 20 points: `[0, 0, 32]` (0x00, 0x00, 0x20)
    - 100 points: `[0, 1, 0]` (0x00, 0x01, 0x00)
    - 10000 points: `[1, 0, 0]` (0x01, 0x00, 0x00)

### Lives & Divers
- **RAM[59]**: Lives/Reserve Subs (0-3)
  - Starts at 3
  - Decrements on death
  - Game over when reaches 0

- **RAM[61]**: Divers Saved (banked at surface)
  - Increments when surfacing with divers
  - Permanent progress metric

- **RAM[62]**: Divers Currently Onboard (0-6 max)
  - Increments when picking up diver
  - Decrements/resets when surfacing
  - Can lose divers if surface too early

### Player State
- **RAM[70]**: Player X position (horizontal)
  - Increases when moving right
  - Decreases when moving left
  - Outmost positions of player sub: left=21, right=134

- **RAM[97]**: Player Y position (vertical)
  - Lower values = higher on screen (closer to surface)
  - Higher values = deeper underwater
  - Surface = 0-13, deepest the player can go: 108

- **RAM[102]**: Oxygen level
  - Starts at full (64+) when at surface
  - Decreases over time when underwater
  - Refills when at surface (Y <= 13) or slightly below the surface (ca.13-15)
  - Death when reaches 0

- **RAM[105]**: Death Animation Timer
  - 0 = Normal play
  - 20 → 0 = Explosion/death sequence countdown
  - Episode resets when reaches 0

---

## Object Tracking

### Enemy/Object Positions (RAM[30-34])
**5 dynamic slots that get reassigned based on spawned objects**

- **RAM[30-34]**: X positions for up to 5 simultaneous enemies/objects
  - **Value = 0**: Slot inactive/empty
  - **Value > 0**: Object present at that X coordinate
  

### Object Type Markers (RAM[36-43])

- **RAM[36-39]**: Primary object type flags
  - `[4, 1, 1, 1]` = Standard enemies (sharks/divers/subs)
  - real meaning still unknown
  - Pattern matches RAM[30-34] slot usage

- **RAM[40-43]**: Secondary object status flags
  - `[4, 1, 1, 1]` = Objects active
  - `[0, 0, 0, 0]` = All objects cleared/despawned
  - Used to track active vs inactive slots

**Despawn detection:**
- When object killed: corresponding status flag changes → 0
- Position may keep old value or reset to 0

### Sub Presence Indicators (RAM[44-47])

- **RAM[44-47]**: Enemy submarine detection
  - **Default (no subs)**: `[200, 200, 200, 200]`
  - **Subs present**: `[8, 8, 8, 8]`
  - Number of non-200 values = number of sub slots active

### Torpedo Tracking (RAM[71-74])

- **RAM[71-74]**: Enemy torpedo X positions
  - Up to 4 simultaneous enemy torpedoes
  - **Value = 0**: No torpedo in slot
  - **Value > 0**: Torpedo at that X position
  - Torpedoes move left-to-right

**Note**: RAM[71-74] also used for diver X positions when no torpedoes active

## Event Detection Patterns

### Diver Pickup Event
**Signature:**
- RAM[62] increases by 1 (divers onboard)
- RAM[73] or RAM[74]: value → 0 (diver X position cleared)
- Player X ≈ Diver X position

### Diver Banking Event (Surfacing)
**Signature:**
- RAM[61] increases (divers saved)
- RAM[62] decreases (divers onboard cleared)
- RAM[102] increases (oxygen refilling)
- RAM[97] < 15 (at surface)

### Enemy Kill Event
**Signature:**
- One of RAM[36-43] flags: 1 → 0 (enemy despawned)
- Corresponding RAM[30-34] position may reset or keep value
- Reward +20 is given

### Death Event
**Signature:**
- RAM[105]: 0 → 20 (death timer starts, counts down 20 → 0)
- RAM[59] decrements by 1 when timer reaches 0
- If RAM[59] was 0 before: full episode reset

### Oxygen Depletion Death
**Signature:**
- RAM[102] reaches 0 (oxygen depleted)
- RAM[105]: 0 → 20 (death sequence)
- Death cause: oxygen (vs collision)

### Enemy Spawn Event
**Signature:**
- One of RAM[30-34]: 0 → >0 (position appears)
- Corresponding RAM[36-43] flag: 0 → 1 (activation)
- For subs: RAM[44-47] changes from 200 → 8

### Torpedo Fire Event
**Signature:**
- RAM[71-74]: 0 → X position (torpedo spawns at sub position)

---

## Y-Position Lane System

**Important**: Enemies stay in fixed Y lanes - Y coordinates NOT stored in RAM!

**Lane types** (approximate Y values):
- **Patrol boat lane**: 0
- **Top lane**: 1
- **Mid-upper lane**: 2
- **Mid-lower lane**: 3
- **Bottom lane**: 4

Player can move freely in Y, but enemies are lane-locked.
---

## Usage Notes for Event Recognition

### Multi-Object Tracking
The dynamic slot system means:
1. **Track slot changes**, not absolute positions
2. **Compare frame-to-frame** to detect spawns/despawns
3. **Correlate flags (36-43) with positions (30-34)** to determine object states

### Counting Active Objects
```python
active_enemies = sum(1 for x in ram[30:35] if x > 0)
active_torpedoes = sum(1 for x in ram[71:75] if x > 0)
active_subs = sum(1 for val in ram[44:48] if val == 8)
```

### Movement Direction Detection
```python
# Track X over time
if current_x > previous_x:
    direction = "LEFT_TO_RIGHT"
elif current_x < previous_x:
    direction = "RIGHT_TO_LEFT"
```

### Proximity Detection
```python
# Check if player near enemy
player_x = ram[70]
for enemy_x in ram[30:35]:
    if enemy_x > 0 and abs(player_x - enemy_x) < 20:
        # Collision risk!
```

---

## Validation Status

**Confirmed** (verified through multiple manual tests):
- Score (56-58)
- Extra lives / reserve subs (59)
- Succesfull surfacings with 6 divers (61)
- Divers currently onboard (62) 0-6
- Player x position (70) (leftmost=21, rightmost=134)
- Player y position (97) (at surface=0-13, deepest=108)
- Oxygen (102)
- Death timer (105) counting down from 20 to 0
- Enemy lane status (40-43) active lanes != 0
- Diver lane status (44-47) swimming divers = 1

**Partially Confirmed** (observed but need more testing):
- Object type flags (36-43) - pattern clear but exact meanings uncertain
- Torpedo positions (71-74)

---

## Implementation Example

```python
def get_game_state(ram):
    """Extract full Seaquest game state from RAM."""
    
    # Score (BCD decode)
    score = int(f"{ram[56]:02x}{ram[57]:02x}{ram[58]:02x}")
    
    # Core state
    state = {
        'score': score,
        'lives': ram[59],
        'divers_saved': ram[61],
        'divers_onboard': ram[62],
        'player_x': ram[70],
        'player_y': ram[97],
        'oxygen': ram[102],
        'death_timer': ram[105],
        
        # Object tracking
        'enemy_positions': [ram[i] for i in range(30, 35)],
        'active_enemies': sum(1 for x in ram[30:35] if x > 0),
        'enemy_flags': [ram[i] for i in range(36, 40)],
        
        # Subs & torpedoes
        'subs_present': sum(1 for x in ram[44:48] if x == 8),
        'torpedo_positions': [ram[i] for i in range(71, 75)],
        'active_torpedoes': sum(1 for x in ram[71:75] if x > 0),
        
        # State flags
        'at_surface': ram[97] < 15,
        'is_dying': ram[105] > 0,
        'oxygen_critical': ram[102] < 10,
    }
    
    return state
```
