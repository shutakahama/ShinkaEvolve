# Disaster Rescue Task

Evolve coordination strategies for rescue vehicles in disaster scenarios.

## Overview

This task simulates emergency rescue operations where multiple vehicles must efficiently rescue targets with varying severity levels. The goal is to evolve strategies that minimize rescue time while prioritizing high-severity targets.

## Default environment

- **Grid**: 10×10 cells
- **Vehicles**: 5 rescue vehicles
- **Targets**: 15 targets with severity levels (1-10)
- **Obstacles**: 10 blocked cells
- **add_target_rate**: 0.0 (Static)
- **add_obstacle_rate**: 0.0 (Static)
- **Max Steps**: 20 steps per scenario

## Strategy Function

Evolve the `select_target` function to assign targets to vehicles:

```python
def select_target(
    vehicle_id: int,
    vehicle_pos: tuple[int, int],
    unrescued_targets: list[dict],  # [{'id', 'pos', 'severity'}, ...]
    other_vehicles: list[dict],      # [{'id', 'pos', 'target_id'}, ...]
    grid_size: tuple[int, int],
    obstacles: list[tuple[int, int]],
) -> Optional[int]:  # Returns target_id or None
```

## Evaluation

### Default Scenarios

7 diverse scenarios prevent overfitting:

1. **Random** (×2): Random placement
2. **Clustered Targets**: Targets grouped in corners
3. **Split Targets**: Targets divided left/right
4. **Maze**: Complex obstacle patterns
5. **Clustered Vehicles**: All vehicles start together
6. **Diagonal Wall**: Diagonal barrier divides map

### Scoring

```
score = - Σ(severity × log(time_penalty))
```

- Logarithmic penalty emphasizes early rescues
- Normalized to 0-1 range
- Final score: average across all scenarios

### Dynamic Mode (Optional)

Enable time-based increase of new targets and obstacles:

- New targets/obstacles is added following Poisson distribution
- Adjusts scoring for fairness (counts time from addition)

## Usage

### Basic Evaluation

```bash
# Evaluate initial strategy
python evaluate.py

# With visualization (creates GIF)
python evaluate.py --visualize
```

### Run Evolution

```bash
python run_evo.py
```

Results saved to `results_disaster_rescue/`.

### Output Files

- `metrics.json`: Combined scores and statistics
- `scenarios_detail.json`: Per-scenario results
- `simulation_*.gif`: Animated visualization (if `--visualize`)

## Visualization

GIF animations show:

- **Targets** (circles): Color-coded by severity (yellow→red)
- **Vehicles** (squares): Match assigned target color
- **Arrows**: Show target direction
- **Obstacles** (gray squares): Blocked cells
- **Rescued** (gray circles with ✓): Completed rescues

## Files

```
disaster_rescue/
├── README.md              # This file
├── initial.py            # Baseline strategy
├── evaluate.py           # Evaluation script
├── scenarios.py          # Scenario generators
└── run_evo.py           # Evolution runner
```

## Key Features

- **Simultaneous Movement**: All vehicles move at the same time (realistic)
- **BFS Pathfinding**: Accurate distance calculation around obstacles
- **Multiple Scenarios**: Prevents overfitting to specific layouts
- **Logarithmic Scoring**: Prioritizes early high-severity rescues
- **Dynamic Mode**: Adapts to changing environments


## License

Part of the ShinkaEvolve framework.
