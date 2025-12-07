# EVOLVE-BLOCK-START
"""
Disaster Rescue Task: Target Assignment Strategy

This module implements a strategy for assigning rescue vehicles to targets
in a disaster scenario. The goal is to rescue all targets as quickly as possible,
prioritizing high-severity targets.
"""

from typing import List, Tuple, Dict, Optional


def select_target(
    vehicle_id: int,
    vehicle_pos: Tuple[int, int],
    unrescued_targets: List[Dict],
    other_vehicles: List[Dict],
    grid_size: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
) -> Optional[int]:
    """
    Select which target this vehicle should go to next.
    
    Args:
        vehicle_id: ID of the current vehicle (0-indexed)
        vehicle_pos: Current position of the vehicle as (row, col)
        unrescued_targets: List of dicts with keys:
            - 'id': target ID
            - 'pos': (row, col) position
            - 'severity': severity level (1-10, higher = more urgent)
        other_vehicles: List of dicts with keys:
            - 'id': vehicle ID
            - 'pos': (row, col) position
            - 'target_id': currently assigned target ID (or None)
        grid_size: Tuple (rows, cols) representing grid dimensions
        obstacles: List of (row, col) positions that are blocked
        
    Returns:
        Target ID to pursue, or None if no targets available
    """
    if not unrescued_targets:
        return None
    
    # Simple greedy strategy: go to the nearest unrescued target
    # This is a baseline that evolution will improve upon
    
    min_distance = float('inf')
    best_target_id = None
    
    for target in unrescued_targets:
        target_pos = target['pos']
        # Calculate Manhattan distance
        distance = abs(vehicle_pos[0] - target_pos[0]) + abs(vehicle_pos[1] - target_pos[1])
        
        if distance < min_distance:
            min_distance = distance
            best_target_id = target['id']
    
    return best_target_id


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
