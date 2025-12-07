# EVOLVE-BLOCK-START
from typing import List, Tuple, Dict, Optional, Set
from collections import deque

class ModularAuctionCoordinator:
    """
    Encapsulates the logic for coordinating vehicle-target assignments.

    This class provides a structured, object-oriented approach to the assignment
    problem, breaking it down into logical phases:
    1. Pre-computation of distances and base utilities.
    2. Enhancement of bids using advanced heuristics.
    3. A deterministic auction to finalize assignments.

    Each vehicle instantiates this class to independently compute the same global
    assignment plan, ensuring coordinated action without direct communication.
    """

    def __init__(
        self,
        all_vehicles: List[Dict],
        unrescued_targets: List[Dict],
        grid_size: Tuple[int, int],
        obstacles: List[Tuple[int, int]],
    ):
        self.all_vehicles = all_vehicles
        self.unrescued_targets = unrescued_targets
        self.grid_size = grid_size
        self.obstacles_set = set(obstacles)

        self._path_memo: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        self.assignments: Dict[int, int] = {}

        # Heuristic parameters from top-performing models
        self.SEVERITY_EXPONENT = 3.0
        self.MUTUAL_PREFERENCE_BONUS_FACTOR = 0.2
        self.BASE_ISOLATION_BONUS_FACTOR = 0.1
        self.UNIQUENESS_BONUS_FACTOR = 0.15  # How much uniqueness affects score
        self.MAX_UNIQUENESS_RATIO = 5.0  # Cap for uniqueness bonus
        self.CRITICAL_TARGET_THRESHOLD = len(all_vehicles) + 1 if len(all_vehicles) > 0 else 1 # Robust to 0 vehicles

    def _bfs_path_len(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Calculates shortest path length using BFS with memoization."""
        if start == end:
            return 0

        if (start, end) in self._path_memo:
            return self._path_memo[(start, end)]
        if (end, start) in self._path_memo:
            return self._path_memo[(end, start)]

        if start in self.obstacles_set or end in self.obstacles_set:
            self._path_memo[(start, end)] = float('inf')
            return float('inf')

        q = deque([(start, 0)])
        visited = {start}
        rows, cols = self.grid_size

        while q:
            (r, c), dist = q.popleft()
            if (r, c) == end:
                self._path_memo[(start, end)] = float(dist)
                return float(dist)

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in self.obstacles_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist + 1))

        self._path_memo[(start, end)] = float('inf')
        return float('inf')

    def _precompute_all_pairs_info(self) -> None:
        """
        Calculates all distances, raw utilities, and "best for" relationships.
        This populates instance variables for use in later stages.
        """
        self.distances: Dict[int, Dict[int, float]] = {v['id']: {} for v in self.all_vehicles}
        self.raw_utilities: Dict[int, Dict[int, float]] = {v['id']: {} for v in self.all_vehicles}
        self.reachable_vehicles_count: Dict[int, int] = {t['id']: 0 for t in self.unrescued_targets}

        best_utility_for_target: Dict[int, float] = {t['id']: -1.0 for t in self.unrescued_targets}
        self.best_vehicle_for_target: Dict[int, Optional[int]] = {t['id']: None for t in self.unrescued_targets}

        for vehicle in self.all_vehicles:
            for target in self.unrescued_targets:
                dist = self._bfs_path_len(vehicle['pos'], target['pos'])
                self.distances[vehicle['id']][target['id']] = dist

                if dist != float('inf'):
                    utility = (target['severity'] ** self.SEVERITY_EXPONENT) / (dist + 1)
                    self.raw_utilities[vehicle['id']][target['id']] = utility
                    self.reachable_vehicles_count[target['id']] += 1

                    if utility > best_utility_for_target[target['id']]:
                        best_utility_for_target[target['id']] = utility
                        self.best_vehicle_for_target[target['id']] = vehicle['id']
                else:
                    self.raw_utilities[vehicle['id']][target['id']] = 0.0

        best_utility_for_vehicle: Dict[int, float] = {v['id']: -1.0 for v in self.all_vehicles}
        self.best_target_for_vehicle: Dict[int, Optional[int]] = {v['id']: None for v in self.all_vehicles}
        for v in self.all_vehicles:
            for t in self.unrescued_targets:
                utility = self.raw_utilities[v['id']].get(t['id'], 0.0)
                if utility > best_utility_for_vehicle[v['id']]:
                    best_utility_for_vehicle[v['id']] = utility
                    self.best_target_for_vehicle[v['id']] = t['id']

        # Calculate uniqueness for each target
        self.target_uniqueness_ratios: Dict[int, float] = {t['id']: 1.0 for t in self.unrescued_targets}
        for target in self.unrescued_targets:
            t_id = target['id']

            utilities_for_target = sorted([
                self.raw_utilities[v['id']].get(t_id, 0.0)
                for v in self.all_vehicles if self.raw_utilities[v['id']].get(t_id, 0.0) > 0
            ], reverse=True)

            if len(utilities_for_target) == 1:
                self.target_uniqueness_ratios[t_id] = self.MAX_UNIQUENESS_RATIO
            elif len(utilities_for_target) >= 2:
                # Ratio of best utility to second best, avoids division by zero
                ratio = utilities_for_target[0] / (utilities_for_target[1] + 1e-9)
                self.target_uniqueness_ratios[t_id] = min(ratio, self.MAX_UNIQUENESS_RATIO)

    def _calculate_enhanced_bids(self) -> List[Dict]:
        """
        Creates a list of bids with scores enhanced by heuristics.
        """
        bids = []
        num_targets = len(self.unrescued_targets)

        # --- Adaptive Isolation Bonus Calculation ---
        adaptive_iso_bonus = self.BASE_ISOLATION_BONUS_FACTOR
        if 0 < num_targets <= self.CRITICAL_TARGET_THRESHOLD:
            scaling = (self.CRITICAL_TARGET_THRESHOLD - num_targets) / max(1, self.CRITICAL_TARGET_THRESHOLD - 1)
            adaptive_iso_bonus *= (1 + scaling)

        # --- Center of Mass (COM) Heuristic Calculation ---
        com = (0, 0)
        avg_dist_to_com = 1.0
        if num_targets > 0:
            sum_r = sum(t['pos'][0] for t in self.unrescued_targets)
            sum_c = sum(t['pos'][1] for t in self.unrescued_targets)
            com = (sum_r / num_targets, sum_c / num_targets)

            total_dist_to_com = sum(abs(t['pos'][0] - com[0]) + abs(t['pos'][1] - com[1]) for t in self.unrescued_targets)
            avg_dist_to_com = total_dist_to_com / num_targets if num_targets > 0 else 1.0

        for vehicle in self.all_vehicles:
            for target in self.unrescued_targets:
                score = self.raw_utilities[vehicle['id']].get(target['id'], 0.0)
                distance = self.distances[vehicle['id']].get(target['id'], float('inf'))

                if score <= 0:
                    continue

                # Apply Stacked Uniqueness and Mutual Preference Bonuses
                # Crossover: Apply MUTUAL_PREFERENCE_BONUS_FACTOR for both conditions,
                # and UNIQUENESS_BONUS_FACTOR additionally when applicable.
                
                # Bonus if this vehicle is the best for this target.
                # This combines mutual preference with the uniqueness aspect.
                if vehicle['id'] == self.best_vehicle_for_target.get(target['id']):
                    score *= (1.0 + self.MUTUAL_PREFERENCE_BONUS_FACTOR) # Base mutual pref if best for target
                    uniqueness_ratio = self.target_uniqueness_ratios.get(target['id'], 1.0)
                    # Additional bonus based on how uniquely good this vehicle is for this target
                    score *= (1.0 + self.UNIQUENESS_BONUS_FACTOR * (uniqueness_ratio - 1.0))

                # Bonus if this target is the best one for this vehicle.
                if target['id'] == self.best_target_for_vehicle.get(vehicle['id']):
                    score *= (1.0 + self.MUTUAL_PREFERENCE_BONUS_FACTOR) # Additional mutual pref if target is best for vehicle

                # Apply Center of Mass (COM) Factor
                if num_targets > 0:
                    dist_to_com = abs(target['pos'][0] - com[0]) + abs(target['pos'][1] - com[1])
                    com_factor = (avg_dist_to_com + 1) / (dist_to_com + 1)
                    dampening_weight = (10.0 - target['severity']) / 10.0
                    dampened_com_factor = (1.0 - dampening_weight) + dampening_weight * com_factor
                    score *= dampened_com_factor

                # Apply Adaptive Isolation Bonus
                num_reachable = self.reachable_vehicles_count[target['id']]
                if len(self.all_vehicles) > 1 and num_reachable > 0:
                    iso_multiplier = 1.0 + adaptive_iso_bonus * \
                        (len(self.all_vehicles) - num_reachable) / (len(self.all_vehicles) - 1)
                    score *= iso_multiplier

                bids.append({
                    'score': score,
                    'distance': distance,
                    'vehicle_id': vehicle['id'],
                    'target_id': target['id'],
                })
        return bids

    def _run_auction(self, bids: List[Dict]) -> None:
        """
        Sorts bids and performs a deterministic greedy assignment.
        """
        bids.sort(key=lambda x: (-x['score'], x['distance'], x['vehicle_id'], x['target_id']))

        assigned_targets = set()
        assigned_vehicles = set()

        for bid in bids:
            v_id, t_id = bid['vehicle_id'], bid['target_id']
            if v_id not in assigned_vehicles and t_id not in assigned_targets:
                self.assignments[v_id] = t_id
                assigned_vehicles.add(v_id)
                assigned_targets.add(t_id)

    def determine_assignments(self) -> Dict[int, int]:
        """
        Orchestrates the entire assignment process and returns the final plan.
        """
        if not self.unrescued_targets:
            return {}

        self._precompute_all_pairs_info()
        bids = self._calculate_enhanced_bids()
        self._run_auction(bids)
        return self.assignments


def select_target(
    vehicle_id: int,
    vehicle_pos: Tuple[int, int],
    unrescued_targets: List[Dict],
    other_vehicles: List[Dict],
    grid_size: Tuple[int, int],
    obstacles: List[Tuple[int, int]],
) -> Optional[int]:
    """
    Selects a target by instantiating and running the ModularAuctionCoordinator.

    This function acts as a public interface to the more complex, class-based
    coordination logic. It constructs the full state of the system and uses the
    coordinator to derive the optimal assignment for all vehicles, returning the
    one designated for the current vehicle.
    """
    if not unrescued_targets:
        return None

    all_vehicles = sorted([{'id': vehicle_id, 'pos': vehicle_pos}] + other_vehicles, key=lambda v: v['id'])

    coordinator = ModularAuctionCoordinator(
        all_vehicles=all_vehicles,
        unrescued_targets=unrescued_targets,
        grid_size=grid_size,
        obstacles=obstacles
    )

    assignments = coordinator.determine_assignments()

    return assignments.get(vehicle_id)

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
