# EVOLVE-BLOCK-START
"""
Disaster Rescue Task: Target Assignment Strategy

This module implements a strategy for assigning rescue vehicles to targets
in a disaster scenario. The goal is to rescue all targets as quickly as possible,
prioritizing high-severity targets.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

# A persistent BFS cache across all select_target calls within a simulation run.
# This global cache will store computed path lengths, improving performance
# as multiple vehicles will calculate paths to and from similar points.
_bfs_cache_global = {}


def select_target(
    vehicle_id: int,
    vehicle_pos: Tuple[int, int],
    unrescued_targets: List[Dict],
    other_vehicles: List[Dict],
    grid_size: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
) -> Optional[int]:
    """
    Selects a target using a Continuous Adaptive Bidding with Localized Competition strategy.

    This strategy continuously re-evaluates the optimal assignment for all vehicles
    at every time step, allowing vehicles to switch to higher-priority targets.

    1.  **Global Persistent BFS Pathfinding**: A Breadth-First Search (BFS)
        calculates the true path length around obstacles for all possible
        vehicle-target pairs. A global, persistent cache (`_bfs_cache_global`)
        stores these path lengths to avoid redundant computations across
        multiple turns and vehicles.
    2.  **Continuous Adaptive Scoring Parameters**: Exponents for severity and
        distance are dynamically interpolated based on the overall criticality
        of remaining targets and the target-to-vehicle ratio, providing a
        smoother strategic shift than discrete mission phases.
    3.  **Adaptive Competition Penalty with Target-Specific Modulation**: A
        `base_alpha` competition factor is adaptively set, and then for each
        target, this `alpha` is further reduced based on the target's severity.
        This ensures critical targets attract strong bids without excessive
        competition penalties.
    4.  **Dynamic Auction and Greedy Assignment**: All possible assignments (bids)
        are collected, scored with the adaptive parameters and modulated by
        the target-specific competition, and then sorted. A deterministic greedy
        assignment ensures an optimal plan for all vehicles.
    5.  **Preemption**: The entire process runs every turn, allowing dynamic
        reassignment to optimize for new information or changed priorities.
    """
    if not unrescued_targets:
        return None

    obstacles_set = frozenset(obstacles)
    rows, cols = grid_size

    def bfs_path_length(start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Calculates shortest path length using BFS, with a robust, canonical, and persistent cache."""
        if start == end:
            return 0.0

        # Use a canonical key for bidirectionality and include obstacles for dynamic safety.
        cache_key = (tuple(sorted((start, end))), obstacles_set)
        if cache_key in _bfs_cache_global:
            return _bfs_cache_global[cache_key]

        q = deque([(start, 0)])
        visited = {start}

        while q:
            (r, c), dist = q.popleft()

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 4-directional movement
                nr, nc = r + dr, c + dc
                next_pos = (nr, nc)

                if next_pos == end:
                    path_len = float(dist + 1)
                    _bfs_cache_global[cache_key] = path_len
                    return path_len

                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and next_pos not in obstacles_set
                    and next_pos not in visited
                ):
                    visited.add(next_pos)
                    q.append((next_pos, dist + 1))

        # Target is unreachable, cache and return infinity
        _bfs_cache_global[cache_key] = float("inf")
        return float("inf")

    # 1. Collect ALL vehicles for a centralized assignment decision.
    all_vehicles = [{"id": vehicle_id, "pos": vehicle_pos}] + [
        {"id": v["id"], "pos": v["pos"]} for v in other_vehicles
    ]

    # 2. --- Continuous Adaptive Scoring Parameters ---
    num_targets = len(unrescued_targets)
    num_vehicles = len(all_vehicles)
    max_severity_val = 10  # Assuming severity is 1-10

    # Calculate overall criticality and target density for continuous adaptation
    if num_targets > 0:
        avg_severity = sum(t["severity"] for t in unrescued_targets) / num_targets
        # criticality_ratio: how critical are the remaining targets on average? (0.1 to 1.0)
        criticality_ratio = avg_severity / max_severity_val
        target_vehicle_ratio = num_targets / num_vehicles
    else:
        criticality_ratio = 0.0
        target_vehicle_ratio = 0.0

    # severity_exponent: scales up (1.5 to 3.5) with higher average target severity
    severity_exponent = 1.5 + 2.0 * criticality_ratio

    # dist_exponent: scales up (1.0 to 2.0) as targets become scarcer relative to vehicles
    # (i.e., when target_vehicle_ratio is low, focus more on efficiency)
    dist_exponent = 1.0 + 1.0 * (
        1 - min(1.0, max(0.0, target_vehicle_ratio / num_vehicles))
    )  # Max dist_exponent when T/V ratio is 0, min when T/V ratio is >= num_vehicles

    # 3. --- Adaptive Competition Penalty with Target-Specific Modulation ---
    # base_alpha: scales up (0.1 to 0.6) when there are many targets per vehicle (encourage spreading)
    base_alpha = 0.1 + 0.5 * min(
        1.0, max(0.0, target_vehicle_ratio / num_vehicles)
    )  # Max base_alpha when T/V ratio is >= num_vehicles

    raw_bids = []
    for vehicle in all_vehicles:
        for target in unrescued_targets:
            path_length = bfs_path_length(vehicle["pos"], target["pos"])

            if path_length != float("inf"):
                # Apply continuous adaptive scoring
                score = (target["severity"] ** severity_exponent) / (
                    (path_length + 1.0) ** dist_exponent
                )
                raw_bids.append(
                    {
                        "score": score,
                        "vehicle_id": vehicle["id"],
                        "target_id": target["id"],
                        "target_severity": target[
                            "severity"
                        ],  # Store severity for competition modulation
                    }
                )

    if not raw_bids:
        return None  # No reachable targets for any vehicle.

    # Calculate total raw scores per target for competition
    competition_sums = {t["id"]: 0.0 for t in unrescued_targets}
    for bid in raw_bids:
        competition_sums[bid["target_id"]] += bid["score"]

    # 4. Modulate bids based on competition with target-specific alpha.
    final_bids = []
    for bid in raw_bids:
        my_raw_score = bid["score"]
        target_id = bid["target_id"]
        target_severity = bid["target_severity"]

        total_competition_for_target = competition_sums[target_id]
        other_competition = total_competition_for_target - my_raw_score

        # Adjust alpha based on target severity: higher severity = lower competition penalty
        # Reduction factor: ranges from 0 (severity 1) to 0.7 (severity 10)
        severity_reduction_factor = (target_severity / max_severity_val) * 0.7
        alpha_for_target = base_alpha * (1.0 - severity_reduction_factor)

        # Apply competition penalty
        final_score = my_raw_score - alpha_for_target * other_competition

        final_bids.append(
            {
                "score": final_score,
                "vehicle_id": bid["vehicle_id"],
                "target_id": target_id,
            }
        )

    # 5. Sort final bids to find the best assignments (highest score).
    # Tie-break deterministically using vehicle_id then target_id.
    final_bids.sort(
        key=lambda x: (x["score"], -x["vehicle_id"], -x["target_id"]), reverse=True
    )

    # 6. Perform deterministic greedy assignment.
    assignments = {}  # Maps {vehicle_id: target_id}
    assigned_targets = set()
    assigned_vehicles = set()

    for bid in final_bids:
        v_id = bid["vehicle_id"]
        t_id = bid["target_id"]

        if v_id not in assigned_vehicles and t_id not in assigned_targets:
            assignments[v_id] = t_id
            assigned_vehicles.add(v_id)
            assigned_targets.add(t_id)

    # 7. Return the assignment for the current vehicle.
    return assignments.get(vehicle_id, None)


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
