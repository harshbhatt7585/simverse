# Farmtila

Farmtila is a grid-based multi-agent RL environment. Agents explore a large field, gather seeds, and harvest tiles; the agent with the most harvested land wins.

- Seeds spawn randomly across the map at the start of the simulation and again every **X** total agent steps.
- Agents must collect seeds before harvesting; each seed can harvest exactly one grid cell.

## Action Space

Discrete (6)

0. move up
1. move down
2. move left
3. move right
4. pick up a seed
5. harvest the land

## Grid World

- Size: `W × H`

## World Layers

### Seed grid

- `seed_grid[x, y] ∈ {0, 1}` — `1` indicates a seed is present in that cell

### Land ownership

- `owner_grid[x, y] ∈ {-1, 0..N-1}`
- `-1` means unharvested, `i` means harvested by agent `i`

### Agent position

- `pos[i] = (x, y)`

## Observation Space

Each agent observes the full environment state:

1. the current `owner_grid`
2. the current `seed_grid`
3. the agent's own position
4. the agent's inventory count (number of acquired seeds)
