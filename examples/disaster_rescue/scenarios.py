"""
Scenario generation functions for disaster rescue simulation.

This module contains various scenario generators that create different
challenging situations for rescue vehicle coordination.
"""

import numpy as np


def generate_random_scenario(
    grid_size: tuple[int, int],
    num_targets: int,
    num_vehicles: int,
    num_obstacles: int,
    random_state: np.random.RandomState,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """
    Generate completely random positions for targets, vehicles, and obstacles.

    Args:
        grid_size: (rows, cols) grid dimensions
        num_targets: Number of rescue targets
        num_vehicles: Number of rescue vehicles
        num_obstacles: Number of obstacles
        random_state: NumPy random state for reproducibility

    Returns:
        Tuple of (targets, vehicles, obstacles)
    """
    occupied = set()
    targets = []
    vehicles = []
    obstacles = []

    # Generate obstacles
    while len(obstacles) < num_obstacles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            obstacles.append(pos)
            occupied.add(pos)

    # Generate targets with severity levels
    target_id = 0
    while len(targets) < num_targets:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Generate vehicles
    vehicle_id = 0
    while len(vehicles) < num_vehicles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            vehicles.append(
                {
                    "id": vehicle_id,
                    "pos": pos,
                    "target_id": None,
                }
            )
            occupied.add(pos)
            vehicle_id += 1

    return targets, vehicles, obstacles


def generate_clustered_targets_scenario(
    grid_size: tuple[int, int],
    num_targets: int,
    num_vehicles: int,
    num_obstacles: int,
    random_state: np.random.RandomState,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """
    Generate scenario with targets clustered in four corners.
    Tests ability to handle spatially separated target groups.

    Args:
        grid_size: (rows, cols) grid dimensions
        num_targets: Number of rescue targets
        num_vehicles: Number of rescue vehicles
        num_obstacles: Number of obstacles
        random_state: NumPy random state for reproducibility

    Returns:
        Tuple of (targets, vehicles, obstacles)
    """
    occupied = set()
    targets = []
    vehicles = []
    obstacles = []

    # Generate obstacles randomly
    while len(obstacles) < num_obstacles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            obstacles.append(pos)
            occupied.add(pos)

    # Generate targets clustered in four corners
    corners = [
        (0, 0),
        (0, grid_size[1] - 1),
        (grid_size[0] - 1, 0),
        (grid_size[0] - 1, grid_size[1] - 1),
    ]
    target_id = 0
    targets_per_corner = num_targets // 4

    for corner in corners:
        count = 0
        while count < targets_per_corner and target_id < num_targets:
            # Generate near corner (within 3 cells)
            offset_r = random_state.randint(-2, 3)
            offset_c = random_state.randint(-2, 3)
            pos = (
                max(0, min(grid_size[0] - 1, corner[0] + offset_r)),
                max(0, min(grid_size[1] - 1, corner[1] + offset_c)),
            )
            if pos not in occupied:
                severity = random_state.randint(1, 11)
                targets.append(
                    {
                        "id": target_id,
                        "pos": pos,
                        "severity": severity,
                        "rescued": False,
                        "rescue_time": None,
                    }
                )
                occupied.add(pos)
                target_id += 1
                count += 1

    # Fill remaining targets randomly
    while target_id < num_targets:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Generate vehicles in center
    center = (grid_size[0] // 2, grid_size[1] // 2)
    vehicle_id = 0
    while vehicle_id < num_vehicles:
        offset_r = random_state.randint(-2, 3)
        offset_c = random_state.randint(-2, 3)
        pos = (center[0] + offset_r, center[1] + offset_c)
        if (
            0 <= pos[0] < grid_size[0]
            and 0 <= pos[1] < grid_size[1]
            and pos not in occupied
        ):
            vehicles.append(
                {
                    "id": vehicle_id,
                    "pos": pos,
                    "target_id": None,
                }
            )
            occupied.add(pos)
            vehicle_id += 1

    return targets, vehicles, obstacles


def generate_split_targets_scenario(
    grid_size: tuple[int, int],
    num_targets: int,
    num_vehicles: int,
    num_obstacles: int,
    random_state: np.random.RandomState,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """
    Generate scenario with targets split on left and right sides.
    Tests vehicle team splitting and coordination.

    Args:
        grid_size: (rows, cols) grid dimensions
        num_targets: Number of rescue targets
        num_vehicles: Number of rescue vehicles
        num_obstacles: Number of obstacles
        random_state: NumPy random state for reproducibility

    Returns:
        Tuple of (targets, vehicles, obstacles)
    """
    occupied = set()
    targets = []
    vehicles = []
    obstacles = []

    # Generate obstacles randomly
    while len(obstacles) < num_obstacles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            obstacles.append(pos)
            occupied.add(pos)

    # Generate targets on left and right sides
    target_id = 0
    half = num_targets // 2

    # Left side
    while target_id < half:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1] // 3),
        )
        if pos not in occupied:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Right side
    while target_id < num_targets:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(2 * grid_size[1] // 3, grid_size[1]),
        )
        if pos not in occupied:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Generate vehicles in middle
    vehicle_id = 0
    while vehicle_id < num_vehicles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(grid_size[1] // 3, 2 * grid_size[1] // 3),
        )
        if pos not in occupied:
            vehicles.append(
                {
                    "id": vehicle_id,
                    "pos": pos,
                    "target_id": None,
                }
            )
            occupied.add(pos)
            vehicle_id += 1

    return targets, vehicles, obstacles


def generate_maze_scenario(
    grid_size: tuple[int, int],
    num_targets: int,
    num_vehicles: int,
    num_obstacles: int,
    random_state: np.random.RandomState,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """
    Generate scenario with maze-like obstacle patterns.
    Tests pathfinding and detour handling.
    Note: num_obstacles is ignored; obstacles are placed based on grid size.

    Args:
        grid_size: (rows, cols) grid dimensions
        num_targets: Number of rescue targets
        num_vehicles: Number of rescue vehicles
        num_obstacles: Number of obstacles (ignored for this scenario)
        random_state: NumPy random state for reproducibility

    Returns:
        Tuple of (targets, vehicles, obstacles)
    """
    occupied = set()
    targets = []
    vehicles = []
    obstacles = []

    # Create maze-like obstacles (vertical walls with gaps at center)
    # Place walls every 3 rows, covering most columns except center gap
    for i in range(2, grid_size[0] - 2, 3):
        for j in range(1, grid_size[1] - 1):
            # Leave gap in the middle for passage
            if j != grid_size[1] // 2:
                pos = (i, j)
                if pos not in occupied:
                    obstacles.append(pos)
                    occupied.add(pos)

    # Add some horizontal wall segments alternating with vertical ones
    for i in range(grid_size[0]):
        if i % 3 == 1 and i not in range(2, grid_size[0] - 2, 3):
            for j in range(2, grid_size[1] - 2, 3):
                # Leave gap at row center
                if i != grid_size[0] // 2:
                    pos = (i, j)
                    if pos not in occupied:
                        obstacles.append(pos)
                        occupied.add(pos)

    # Generate targets randomly
    target_id = 0
    while target_id < num_targets:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Generate vehicles randomly
    vehicle_id = 0
    while vehicle_id < num_vehicles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            vehicles.append(
                {
                    "id": vehicle_id,
                    "pos": pos,
                    "target_id": None,
                }
            )
            occupied.add(pos)
            vehicle_id += 1

    return targets, vehicles, obstacles


def generate_clustered_vehicles_scenario(
    grid_size: tuple[int, int],
    num_targets: int,
    num_vehicles: int,
    num_obstacles: int,
    random_state: np.random.RandomState,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """
    Generate scenario with vehicles starting clustered together.
    Tests efficient dispersion and coordination from a single location.

    Args:
        grid_size: (rows, cols) grid dimensions
        num_targets: Number of rescue targets
        num_vehicles: Number of rescue vehicles
        num_obstacles: Number of obstacles
        random_state: NumPy random state for reproducibility

    Returns:
        Tuple of (targets, vehicles, obstacles)
    """
    occupied = set()
    targets = []
    vehicles = []
    obstacles = []

    # Generate obstacles randomly
    while len(obstacles) < num_obstacles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            obstacles.append(pos)
            occupied.add(pos)

    # Generate targets randomly
    target_id = 0
    while target_id < num_targets:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Generate vehicles clustered in one corner
    corner = (1, 1)
    vehicle_id = 0
    attempts = 0
    max_attempts = 100
    while vehicle_id < num_vehicles and attempts < max_attempts:
        offset_r = attempts // 3
        offset_c = attempts % 3
        pos = (corner[0] + offset_r, corner[1] + offset_c)
        if (
            0 <= pos[0] < grid_size[0]
            and 0 <= pos[1] < grid_size[1]
            and pos not in occupied
        ):
            vehicles.append(
                {
                    "id": vehicle_id,
                    "pos": pos,
                    "target_id": None,
                }
            )
            occupied.add(pos)
            vehicle_id += 1
        attempts += 1

    # If couldn't place all vehicles in cluster, place rest randomly
    while vehicle_id < num_vehicles:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied:
            vehicles.append(
                {
                    "id": vehicle_id,
                    "pos": pos,
                    "target_id": None,
                }
            )
            occupied.add(pos)
            vehicle_id += 1

    return targets, vehicles, obstacles


def generate_diagonal_wall_scenario(
    grid_size: tuple[int, int],
    num_targets: int,
    num_vehicles: int,
    num_obstacles: int,
    random_state: np.random.RandomState,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """
    Generate scenario with a diagonal wall dividing the map.
    Tests handling of bisected areas requiring large detours.
    Note: num_obstacles is ignored; obstacles are placed based on grid size.

    Args:
        grid_size: (rows, cols) grid dimensions
        num_targets: Number of rescue targets
        num_vehicles: Number of rescue vehicles
        num_obstacles: Number of obstacles (ignored for this scenario)
        random_state: NumPy random state for reproducibility

    Returns:
        Tuple of (targets, vehicles, obstacles)
    """
    occupied = set()
    targets = []
    vehicles = []
    obstacles = []

    # Create diagonal wall - main diagonal
    for i in range(grid_size[0]):
        j = i
        if j < grid_size[1]:
            pos = (i, j)
            obstacles.append(pos)
            occupied.add(pos)

    # Add adjacent diagonal to make wall thicker and more challenging
    for i in range(grid_size[0] - 1):
        j = i + 1
        if j < grid_size[1]:
            pos = (i, j)
            obstacles.append(pos)
            occupied.add(pos)

    # Add some perpendicular segments to create bottlenecks
    # Add segments at 1/3 and 2/3 positions
    third_row = grid_size[0] // 3
    two_third_row = 2 * grid_size[0] // 3

    for offset in [-1, 0, 1]:
        # Upper segment
        r = third_row
        c = third_row + offset
        if 0 <= r < grid_size[0] and 0 <= c < grid_size[1]:
            pos = (r, c)
            if pos not in occupied:
                obstacles.append(pos)
                occupied.add(pos)

        # Lower segment
        r = two_third_row
        c = two_third_row + offset
        if 0 <= r < grid_size[0] and 0 <= c < grid_size[1]:
            pos = (r, c)
            if pos not in occupied:
                obstacles.append(pos)
                occupied.add(pos)

    # Generate targets on opposite sides of diagonal
    target_id = 0
    half = num_targets // 2

    # Upper-left side
    while target_id < half:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied and pos[1] < pos[0]:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Lower-right side
    while target_id < num_targets:
        pos = (
            random_state.randint(0, grid_size[0]),
            random_state.randint(0, grid_size[1]),
        )
        if pos not in occupied and pos[1] > pos[0]:
            severity = random_state.randint(1, 11)
            targets.append(
                {
                    "id": target_id,
                    "pos": pos,
                    "severity": severity,
                    "rescued": False,
                    "rescue_time": None,
                }
            )
            occupied.add(pos)
            target_id += 1

    # Generate vehicles near center
    center = (grid_size[0] // 2, grid_size[1] // 2)
    vehicle_id = 0
    while vehicle_id < num_vehicles:
        offset_r = random_state.randint(-2, 3)
        offset_c = random_state.randint(-2, 3)
        pos = (center[0] + offset_r, center[1] + offset_c)
        if (
            0 <= pos[0] < grid_size[0]
            and 0 <= pos[1] < grid_size[1]
            and pos not in occupied
        ):
            vehicles.append(
                {
                    "id": vehicle_id,
                    "pos": pos,
                    "target_id": None,
                }
            )
            occupied.add(pos)
            vehicle_id += 1

    return targets, vehicles, obstacles


# Scenario registry
SCENARIO_GENERATORS = {
    "random": generate_random_scenario,
    "clustered_targets": generate_clustered_targets_scenario,
    "split_targets": generate_split_targets_scenario,
    "maze": generate_maze_scenario,
    "clustered_vehicles": generate_clustered_vehicles_scenario,
    "diagonal_wall": generate_diagonal_wall_scenario,
}


def get_scenario_generator(scenario_type: str):
    """Get the scenario generator function for a given scenario type."""
    if scenario_type not in SCENARIO_GENERATORS:
        raise ValueError(
            f"Unknown scenario type: {scenario_type}. "
            f"Available types: {list(SCENARIO_GENERATORS.keys())}"
        )
    return SCENARIO_GENERATORS[scenario_type]
