# Disaster Rescue Task

Evolve coordination strategies for rescue vehicles in disaster scenarios.

## Overview

This task simulates emergency rescue operations where multiple vehicles must efficiently rescue targets with varying severity levels. The goal is to evolve strategies that minimize rescue time while prioritizing high-severity targets.

## Default environment

- **Grid**: 30×30 cells (configurable)
- **Vehicles**: 5 rescue vehicles
- **Targets**: 15 targets with severity levels (1-10)
- **Obstacles**: 45 blocked cells
- **Max Steps**: 150 steps per scenario

## Strategy Function

Evolve the `select_target` function to assign targets to vehicles:

```python
def select_target(
    vehicle_id: int,
    vehicle_pos: Tuple[int, int],
    unrescued_targets: List[Dict],  # [{'id', 'pos', 'severity'}, ...]
    other_vehicles: List[Dict],      # [{'id', 'pos', 'target_id'}, ...]
    grid_size: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
) -> Optional[int]:  # Returns target_id or None
```

## Evaluation

### Scenarios

7 diverse scenarios prevent overfitting:

1. **Random** (×2): Random placement
2. **Clustered Targets**: Targets grouped in corners
3. **Split Targets**: Targets divided left/right
4. **Maze**: Complex obstacle patterns
5. **Clustered Vehicles**: All vehicles start together
6. **Diagonal Wall**: Diagonal barrier divides map

### Scoring

```
score = Σ(severity × log(max_steps - rescue_time))
```

- Logarithmic penalty emphasizes early rescues
- Normalized to 0-1 range
- Final score: average across all scenarios

### Dynamic Mode (Optional)

Enable time-based spawning of targets and obstacles:

```bash
python evaluate.py --dynamic --visualize
```

- New targets/obstacles spawn following Poisson distribution
- Spawn rates: `--target_spawn_rate 0.05` (default)
- Adjusts scoring for fairness (counts time from spawn)

## Usage

### Basic Evaluation

```bash
# Evaluate initial strategy
python evaluate.py

# With visualization (creates GIF)
python evaluate.py --visualize

# Dynamic mode
python evaluate.py --dynamic --visualize

# Custom scenario count
python evaluate.py --num_scenarios 10
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
- `rescue_log_scenario0.json`: Detailed rescue log

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

## Tips for Evolution

Effective strategies consider:

1. **Coordination**: Avoid multiple vehicles targeting same location
2. **Severity Weighting**: Prioritize high-severity targets
3. **Distance Estimation**: Account for obstacles in path planning
4. **Dynamic Reassignment**: Adapt as situation changes
5. **Global Optimization**: Consider all vehicle-target pairs

## License

Part of the ShinkaEvolve framework.
