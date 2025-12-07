# EVOLVE-BLOCK-START
"""
Disaster Rescue Task: Target Assignment Strategy

This module implements a strategy for assigning rescue vehicles to targets
in a disaster scenario. The goal is to rescue all targets as quickly as possible,
prioritizing high-severity targets.
"""

from typing import List, Tuple, Dict, Optional

# Constants
OBSTACLE_PENALTY = 2  # A more realistic penalty, equal to the minimum detour cost.
# DISTANCE_BUCKET_SIZE removed as bucketing proved inefficient and increased mission time.


def _manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculates Manhattan distance between two points."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def _is_path_obstructed_heuristic(
    start: Tuple[int, int], end: Tuple[int, int], obstacle_set: set
) -> bool:
    """
    Heuristically checks if a direct Manhattan path between start and end
    is blocked by any obstacle. This is a fast check, not a full pathfinding.
    An obstacle is on the path if dist(start, obs) + dist(obs, end) == dist(start, end).
    """
    if not obstacle_set:
        return False

    base_dist = _manhattan_distance(start, end)
    for obs_pos in obstacle_set:
        # Check if obstacle lies on any shortest path between start and end using Manhattan property
        if _manhattan_distance(start, obs_pos) + _manhattan_distance(obs_pos, end) == base_dist:
            return True
    return False


def select_target(
    vehicle_id: int,
    vehicle_pos: Tuple[int, int],
    unrescued_targets: List[Dict],
    other_vehicles: List[Dict],
    grid_size: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
) -> Optional[int]:
    """
    Selects a target using a globally coordinated assignment with distance bucketing.

    This strategy improves upon a strictly distance-first approach by introducing
    "distance buckets." This allows high-severity targets to be prioritized if they
    are "close enough" to the nearest target, providing a better balance between
    minimizing travel time and maximizing rescue urgency.

    The process is as follows:
    1.  **Global Coordination**: A complete list of all vehicles and all unrescued
        targets is compiled for centralized planning.
    2.  **Hierarchical Cost Calculation**: For each (vehicle, target) pair, a
        multi-part cost tuple is created:
        `(distance_bucket, -severity, effective_distance, v_id, t_id)`.
        - `distance_bucket`: `effective_distance // BUCKET_SIZE`. This groups
          targets into coarse proximity bands, making them comparable.
        - `-severity`: Within a bucket, higher severity targets are prioritized.
        - `effective_distance`: If severity is also tied, the closer target is chosen.
        - This structure allows for minor, intelligent detours for high-value
          targets without compromising overall efficiency.
    3.  **Deterministic Sorting**: All potential assignments are sorted based on this
        new cost tuple.
    4.  **Single-Pass Greedy Assignment**: The sorted list is traversed to make
        conflict-free assignments.
    5.  **Action**: The vehicle returns the target assigned to it in the global plan.
    """
    if not unrescued_targets:
        return None

    # Use all vehicles for a global, dynamic replanning at every step.
    all_vehicles = [{'id': vehicle_id, 'pos': vehicle_pos}] + other_vehicles
    obstacle_set = set(obstacles)

    potential_assignments = []
    for vehicle in all_vehicles:
        v_pos = vehicle['pos']
        for target in unrescued_targets:
            t_pos = target['pos']

            base_dist = _manhattan_distance(v_pos, t_pos)

            penalty = 0
            if _is_path_obstructed_heuristic(v_pos, t_pos, obstacle_set):
                penalty = OBSTACLE_PENALTY
            effective_distance = base_dist + penalty

            # Create a composite cost tuple. Lower is better.
            # 1. Primary: Effective distance (incorporating a tuned obstacle penalty).
            # 2. Secondary: Severity, to prioritize more urgent targets when distances are comparable.
            # 3. Tertiary/Quaternary: IDs for deterministic tie-breaking.
            potential_assignments.append(
                (effective_distance, -target['severity'], vehicle['id'], target['id'])
            )

    # Sort all possible assignments by the cost tuple.
    potential_assignments.sort()

    # Perform a greedy assignment based on the sorted list.
    assigned_vehicles = set()
    assigned_targets = set()
    final_assignments = {}

    # Unpack the 4-element cost tuple to perform the assignment.
    for eff_dist, neg_severity, v_id, t_id in potential_assignments:
        if v_id not in assigned_vehicles and t_id not in assigned_targets:
            final_assignments[v_id] = t_id
            assigned_vehicles.add(v_id)
            assigned_targets.add(t_id)

    # Return the target assigned to this specific vehicle.
    return final_assignments.get(vehicle_id, None)

# EVOLVE-BLOCK-END


def run_rescue_simulation(
    vehicle_id: int,
    vehicle_pos: Tuple[int, int],
    unrescued_targets: List[Dict],
    other_vehicles: List[Dict],
    grid_size: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
) -> Optional[int]:
    """
    Wrapper function that calls the evolved select_target function.
    This function signature is what the evaluator will call.
    """
    return select_target(
        vehicle_id=vehicle_id,
        vehicle_pos=vehicle_pos,
        unrescued_targets=unrescued_targets,
        other_vehicles=other_vehicles,
        grid_size=grid_size,
        obstacles=obstacles,
    )
