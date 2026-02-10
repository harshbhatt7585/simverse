# Farmtila

Farmtila is a grid-based multi-agent RL environment where agents collect seeds, build territory, unlock 3x3 regions, and then harvest unlocked land.

- Seeds spawn randomly across the map at the start of the simulation and again every **X** total agent steps.
- Agents automatically collect seeds by stepping on them.
- Land progression has two stages:
1. Claim territory cells (locked) by spending seeds.
2. Complete a 3x3 claimed block to unlock it, then spend more seeds to harvest unlocked cells.

## Action Space

Discrete (6)

0. move up
1. move down
2. move left
3. move right
4. land action (claim territory / harvest unlocked land)
5. optional pickup/no-op (seeds are still auto-collected on entry)

## Grid World

- Size: `W × H`

## World Layers

### Seed grid

- `seed_grid[x, y] ∈ {0, 1}` — `1` indicates a seed is present in that cell

### Land ownership

- `owner_grid[x, y] ∈ {-1, 0..N-1}`
- `-1` means unclaimed, `i` means the cell belongs to agent `i`

### Land state

- `farm_grid[x, y]` uses layered land states:
1. `0` = empty
2. `1` = territory claimed (locked)
3. `2` = territory unlocked (part of a completed 3x3 block)
4. `3` = harvested

### Agent position

- `pos[i] = (x, y)`

## Observation Space

Each agent observes the full environment state (as a 5-channel grid):

1. `seed_grid`
2. `owner_grid`
3. `farm_grid` (land stage: empty/locked/unlocked/harvested)
4. `agent_grid` (agent positions)
5. `inventory_grid` (inventory values at agent positions)


## Rewards
1. Seed pickup: `+1.0`
2. Seed proximity shaping: small reward each step based on closeness to nearest seed
3. Step cost: `-0.005`
4. Claim empty land with seed: `+0.1`
5. Complete and unlock a full 3x3 owned territory block: `+5.0`
6. Spend seed on unlocked owned tile to harvest it: `+1.0`
7. If a newly claimed territory cell is adjacent to already harvested owned land, step cost is waived for that action.
8. First agent to harvest `3` land tiles wins and gets `+50.0`.

## Win and Draw Conditions

- Win: first agent that reaches `harvest_goal` (default `3`) harvested tiles.
- Draw: if no agent reaches the goal before episode end (`max_steps` or seed budget exhaustion).
