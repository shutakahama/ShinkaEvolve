"""
Evaluator for Disaster Rescue Task

This module simulates a disaster rescue scenario where multiple vehicles
must coordinate to rescue targets with varying severity levels.
"""

import argparse
import importlib.util
import io
import json
import os
from collections import deque
from typing import Any, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scenarios import get_scenario_generator

from shinka.core.wrap_eval import save_json_results


class RescueSimulator:
    """Simulates the disaster rescue scenario."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        num_targets: int,
        num_vehicles: int,
        num_obstacles: int,
        random_seed: int,
        scenario_type: str,
        add_target_rate: float,
        add_obstacle_rate: float,
    ):
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.num_vehicles = num_vehicles
        self.num_obstacles = num_obstacles
        self.random_seed = random_seed
        self.scenario_type = scenario_type

        # Dynamic disaster mode parameters
        # add_rate means lambda for poisson (avg per step)
        self.add_target_rate = add_target_rate
        self.add_obstacle_rate = add_obstacle_rate
        self.next_target_id = 0  # Track next ID for dynamically added targets

        np.random.seed(random_seed)
        self.random_state = np.random.RandomState(random_seed)

        # Initialize grid
        self.targets = []
        self.vehicles = []
        self.obstacles = []
        self.rescued_targets = set()

        # Generate scenario
        self._generate_scenario()

    def _generate_scenario(self):
        """Generate scenario using external scenario generator functions."""
        # Get the appropriate scenario generator function
        generator_fn = get_scenario_generator(self.scenario_type)

        # Create random state for reproducibility
        random_state = np.random.RandomState(self.random_seed)

        # Generate scenario
        targets, vehicles, obstacles = generator_fn(
            grid_size=self.grid_size,
            num_targets=self.num_targets,
            num_vehicles=self.num_vehicles,
            num_obstacles=self.num_obstacles,
            random_state=random_state,
        )

        # Set the generated scenario
        self.targets = targets
        self.vehicles = vehicles
        self.obstacles = obstacles

        # Set next_target_id based on initial targets
        self.next_target_id = len(self.targets)

    def _get_occupied_positions(self) -> set:
        """Get all currently occupied positions on the grid."""
        occupied = set()

        # Add vehicle positions
        for vehicle in self.vehicles:
            occupied.add(vehicle["pos"])

        # Add target positions (including rescued ones)
        for target in self.targets:
            occupied.add(target["pos"])

        # Add obstacle positions
        for obstacle in self.obstacles:
            occupied.add(obstacle)

        return occupied

    def _add_targets(self, step: int) -> list[dict]:
        """Add new targets based on Poisson process."""
        if self.add_target_rate <= 0.0:
            return []

        # Number of new targets follows Poisson distribution
        num_new_targets = self.random_state.poisson(self.add_target_rate)

        if num_new_targets == 0:
            return []

        occupied = self._get_occupied_positions()
        new_targets = []

        for _ in range(num_new_targets):
            # Try to find an unoccupied position
            max_attempts = 50
            for attempt in range(max_attempts):
                pos = (
                    self.random_state.randint(0, self.grid_size[0]),
                    self.random_state.randint(0, self.grid_size[1]),
                )

                if pos not in occupied:
                    severity = self.random_state.randint(1, 11)
                    new_target = {
                        "id": self.next_target_id,
                        "pos": pos,
                        "severity": severity,
                        "rescued": False,
                        "rescue_time": None,
                        "added_step": step,  # Track when this target appeared
                    }
                    new_targets.append(new_target)
                    occupied.add(pos)
                    self.next_target_id += 1
                    break

        return new_targets

    def _add_obstacles(self) -> list[tuple[int, int]]:
        """Add new obstacles based on Poisson process."""
        if self.add_obstacle_rate <= 0.0:
            return []

        # Number of new obstacles follows Poisson distribution
        num_new_obstacles = self.random_state.poisson(self.add_obstacle_rate)

        if num_new_obstacles == 0:
            return []

        occupied = self._get_occupied_positions()
        new_obstacles = []

        for _ in range(num_new_obstacles):
            # Try to find an unoccupied position
            max_attempts = 50
            for attempt in range(max_attempts):
                pos = (
                    self.random_state.randint(0, self.grid_size[0]),
                    self.random_state.randint(0, self.grid_size[1]),
                )

                if pos not in occupied:
                    new_obstacles.append(pos)
                    occupied.add(pos)
                    break

        return new_obstacles

    def _find_path(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> Optional[list[tuple[int, int]]]:
        """Find shortest path using BFS, avoiding obstacles."""
        if start == goal:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()

            # Check all 4 directions
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (pos[0] + dr, pos[1] + dc)

                # Check bounds
                if not (
                    0 <= new_pos[0] < self.grid_size[0]
                    and 0 <= new_pos[1] < self.grid_size[1]
                ):
                    continue

                # Check obstacles and visited
                if new_pos in self.obstacles or new_pos in visited:
                    continue

                new_path = path + [new_pos]

                if new_pos == goal:
                    return new_path

                queue.append((new_pos, new_path))
                visited.add(new_pos)

        return None  # No path found

    def _create_visualization_frame(self, step: int) -> plt.Figure:
        """
        Create a single frame visualization of the current simulation state.

        Args:
            step: Current simulation step

        Returns:
            matplotlib Figure object
        """
        # Close any existing figures to prevent memory leaks
        plt.close("all")

        fig, ax = plt.subplots(figsize=(10, 10))

        # Set up the grid
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Column", fontsize=12)
        ax.set_ylabel("Row", fontsize=12)
        ax.set_title(
            f"Disaster Rescue Simulation - Step {step}", fontsize=14, fontweight="bold"
        )

        # Invert y-axis so (0,0) is at top-left
        ax.invert_yaxis()

        # Create color map for targets based on severity
        severity_cmap = plt.colormaps.get_cmap("YlOrRd")  # Yellow to Red colormap

        # Draw obstacles
        for obs_pos in self.obstacles:
            rect = mpatches.Rectangle(
                (obs_pos[1] - 0.4, obs_pos[0] - 0.4),
                0.8,
                0.8,
                facecolor="gray",
                edgecolor="black",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Draw targets
        target_colors = {}
        for target in self.targets:
            if not target["rescued"]:
                severity = target["severity"]
                color = severity_cmap(severity / 10.0)
                target_colors[target["id"]] = color

                # Draw target as a circle
                circle = mpatches.Circle(
                    (target["pos"][1], target["pos"][0]),
                    radius=0.35,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.7,
                )
                ax.add_patch(circle)

                # Add severity label
                ax.text(
                    target["pos"][1],
                    target["pos"][0],
                    str(severity),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                )
            else:
                # Draw rescued targets as gray circles with checkmark
                circle = mpatches.Circle(
                    (target["pos"][1], target["pos"][0]),
                    radius=0.35,
                    facecolor="lightgray",
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.5,
                )
                ax.add_patch(circle)
                ax.text(
                    target["pos"][1],
                    target["pos"][0],
                    "✓",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="green",
                )

        # Draw vehicles
        for vehicle in self.vehicles:
            # Determine vehicle color based on target
            if vehicle["target_id"] is not None and vehicle["target_id"] in target_colors:
                vehicle_color = target_colors[vehicle["target_id"]]
            else:
                vehicle_color = "blue"

            # Draw vehicle as a square
            rect = mpatches.Rectangle(
                (vehicle["pos"][1] - 0.3, vehicle["pos"][0] - 0.3),
                0.6,
                0.6,
                facecolor=vehicle_color,
                edgecolor="black",
                linewidth=2.5,
                alpha=0.9,
            )
            ax.add_patch(rect)

            # Add vehicle ID
            ax.text(
                vehicle["pos"][1],
                vehicle["pos"][0],
                f'V{vehicle["id"]}',
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

            # Draw arrow to target if assigned
            if vehicle["target_id"] is not None:
                target = self.targets[vehicle["target_id"]]
                if not target["rescued"]:
                    ax.arrow(
                        vehicle["pos"][1],
                        vehicle["pos"][0],
                        target["pos"][1] - vehicle["pos"][1],
                        target["pos"][0] - vehicle["pos"][0],
                        head_width=0.15,
                        head_length=0.15,
                        fc=vehicle_color,
                        ec="black",
                        alpha=0.4,
                        linewidth=1.5,
                        length_includes_head=True,
                    )

        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor="gray", edgecolor="black", label="Obstacle"),
            mpatches.Circle(
                (0, 0),
                radius=0.1,
                facecolor=severity_cmap(1.0),
                edgecolor="black",
                label="High Severity Target",
            ),
            mpatches.Circle(
                (0, 0),
                radius=0.1,
                facecolor=severity_cmap(0.1),
                edgecolor="black",
                label="Low Severity Target",
            ),
            mpatches.Circle(
                (0, 0),
                radius=0.1,
                facecolor="lightgray",
                edgecolor="black",
                label="Rescued Target",
            ),
            mpatches.Rectangle(
                (0, 0), 1, 1, facecolor="blue", edgecolor="black", label="Rescue Vehicle"
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            framealpha=0.9,
        )

        # Add statistics
        rescued_count = len([t for t in self.targets if t["rescued"]])
        total_targets = len(self.targets)
        stats_text = f"Rescued: {rescued_count}/{total_targets}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        return fig

    def run_simulation(
        self, strategy_fn, max_steps: int, save_visualization: bool, results_dir: str
    ) -> dict[str, Any]:
        """
        Run the rescue simulation using the provided strategy function.

        Args:
            strategy_fn: Function that takes vehicle info and returns target ID
            max_steps: Maximum simulation steps
            save_visualization: Whether to save visualization frames and create GIF
            results_dir: Directory to save visualization (required if save_visualization=True)

        Returns:
            Dictionary with simulation results and metrics
        """
        step = 0
        rescue_log = []
        added_log = []  # Track dynamically added targets and obstacles
        frames = []  # Store frames for GIF creation

        # Save initial state if visualization is enabled
        if save_visualization:
            if results_dir is None:
                raise ValueError(
                    "results_dir must be provided when save_visualization=True"
                )
            frames.append(self._create_visualization_frame(step))

        # In dynamic situation, continue until all targets are rescued (no target limit)
        # In static mode, use the original target count
        initial_target_count = len(self.targets)

        while step < max_steps:
            step += 1

            # Add dynamic targets and obstacles at the start of this step
            new_targets = self._add_targets(step)
            if new_targets:
                self.targets.extend(new_targets)
                added_log.append(
                    {
                        "step": step,
                        "type": "targets",
                        "count": len(new_targets),
                        "items": [
                            {"id": t["id"], "pos": t["pos"], "severity": t["severity"]}
                            for t in new_targets
                        ],
                    }
                )
                if save_visualization:
                    print(f"  [Step {step}] +{len(new_targets)} new target(s) appeared!")

            new_obstacles = self._add_obstacles()
            if new_obstacles:
                self.obstacles.extend(new_obstacles)
                added_log.append(
                    {
                        "step": step,
                        "type": "obstacles",
                        "count": len(new_obstacles),
                        "positions": new_obstacles,
                    }
                )
                if save_visualization:
                    print(
                        f"  [Step {step}] +{len(new_obstacles)} new obstacle(s) appeared!"
                    )

            # Get unrescued targets (same for all vehicles in this step)
            unrescued_targets = [t for t in self.targets if not t["rescued"]]

            # Check termination condition
            if not unrescued_targets:
                break

            # Record all current vehicle positions at the start of this step
            current_positions = {v["id"]: v["pos"] for v in self.vehicles}

            # Compute target selection for ALL vehicles simultaneously
            vehicle_decisions = []
            for vehicle in self.vehicles:
                # Get other vehicles info (with positions from start of this step)
                other_vehicles = [
                    {
                        "id": v["id"],
                        "pos": current_positions[v["id"]],
                        "target_id": v["target_id"],
                    }
                    for v in self.vehicles
                    if v["id"] != vehicle["id"]
                ]

                # Call strategy function for this vehicle
                try:
                    selected_target_id = strategy_fn(
                        vehicle_id=vehicle["id"],
                        vehicle_pos=current_positions[vehicle["id"]],
                        unrescued_targets=unrescued_targets,
                        other_vehicles=other_vehicles,
                        grid_size=self.grid_size,
                        obstacles=self.obstacles,
                    )
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Strategy function error: {str(e)}",
                        "step": step,
                    }

                vehicle_decisions.append((vehicle, selected_target_id))

            # Now execute all decisions: move vehicles and check for rescues
            for vehicle, selected_target_id in vehicle_decisions:
                if selected_target_id is not None:
                    target = self.targets[selected_target_id]
                    if not target["rescued"]:
                        # Find path to the selected target
                        path = self._find_path(vehicle["pos"], target["pos"])
                        if path is not None and len(path) > 1:
                            # Move one step towards target
                            vehicle["pos"] = path[1]
                            vehicle["target_id"] = selected_target_id

                            # Check if reached target after moving
                            if vehicle["pos"] == target["pos"]:
                                target["rescued"] = True
                                target["rescue_time"] = step
                                self.rescued_targets.add(selected_target_id)
                                rescue_log.append(
                                    {
                                        "step": step,
                                        "vehicle_id": vehicle["id"],
                                        "target_id": selected_target_id,
                                        "severity": target["severity"],
                                    }
                                )
                                vehicle["target_id"] = None

            # Save frame after this step if visualization is enabled
            if save_visualization:
                frames.append(self._create_visualization_frame(step))

        # Create GIF from frames if visualization is enabled
        if save_visualization and frames:
            try:
                file_name = f"simulation_{self.scenario_type}_{self.random_seed}.gif"
                gif_path = os.path.join(results_dir, file_name)
                print(f"Creating visualization GIF with {len(frames)} frames...")

                # Save frames as GIF using imageio or PIL
                images = []
                for frame_fig in frames:
                    # Convert figure to image
                    buf = io.BytesIO()
                    frame_fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
                    buf.seek(0)
                    img = Image.open(buf)
                    images.append(img.copy())
                    buf.close()
                    plt.close(frame_fig)

                # Save as GIF
                duration = 500 if len(images) <= 20 else 300  # milliseconds per frame
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=0,
                )

                print(f"✓ Visualization saved to {gif_path}")
            except Exception as e:
                print(f"Warning: Failed to create visualization GIF: {e}")

        # Calculate metrics
        total_rescued = len(self.rescued_targets)
        total_targets = len(self.targets)
        all_rescued = total_rescued == total_targets

        # Calculate weighted score based on severity and rescue time
        # Higher severity targets rescued earlier get higher scores
        # Use logarithmic time penalty to make early rescues more valuable
        weighted_score = 0
        max_possible_score = 0

        for target in self.targets:
            severity = target["severity"]
            target_max_steps = max_steps - (
                target.get("added_step", 0) if self.add_target_rate > 0.0 else 0
            )
            max_possible_score += severity * np.log(target_max_steps + 1)

            if target["rescued"]:
                # Logarithmic time penalty
                rescue_time = target["rescue_time"] - (
                    target.get("added_step", 0) if self.add_target_rate > 0.0 else 0
                )
                time_penalty = np.log(target_max_steps + 1) - np.log(rescue_time + 1)
                weighted_score += severity * time_penalty

        # Normalize score to 0-1 range
        if max_possible_score > 0:
            normalized_score = weighted_score / max_possible_score
        else:
            normalized_score = 0.0

        result = {
            "success": True,
            "total_steps": step,
            "total_rescued": total_rescued,
            "total_targets": total_targets,  # Actual number including added
            "all_rescued": all_rescued,
            "rescue_rate": total_rescued / total_targets if total_targets > 0 else 0,
            "weighted_score": weighted_score,
            "normalized_score": normalized_score,
            "rescue_log": rescue_log,
        }

        if self.add_target_rate > 0.0 or self.add_obstacle_rate > 0.0:
            result["added_log"] = added_log
            result["initial_target_count"] = initial_target_count
            result["added_target_count"] = total_targets - initial_target_count

        return result


def validate_rescue_result(
    result: dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """Validate the rescue simulation result."""
    if not result.get("success", False):
        return False, result.get("error", "Simulation failed")

    # Check if all targets were rescued
    if not result.get("all_rescued", False):
        return (
            True,
            f"Only {result['total_rescued']} out of {result.get('total_targets', 0)} targets rescued",
        )

    return True, "All targets successfully rescued"


def get_scenario_settings(run_index: int) -> tuple[str, int]:
    """Provides keyword arguments for rescue simulation runs with different scenarios."""
    # Define scenario types to test
    scenario_types = [
        "random",
        "random",  # Test random twice with different seeds
        "clustered_targets",
        "split_targets",
        "maze",
        "clustered_vehicles",
        "diagonal_wall",
    ]

    scenario_type = scenario_types[run_index % len(scenario_types)]

    return scenario_type, 42 + run_index


def aggregate_rescue_metrics(
    results: list[dict[str, Any]], results_dir: str
) -> dict[str, Any]:
    """
    Aggregate metrics across multiple rescue simulation runs.

    Args:
        results: List of simulation result dictionaries
        results_dir: Directory to save results

    Returns:
        Aggregated metrics dictionary
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    # Filter successful results
    successful_results = [r for r in results if r.get("success", False)]

    if not successful_results:
        return {"combined_score": 0.0, "error": "No successful results to aggregate"}

    # Calculate average scores across all scenarios
    total_normalized_score = sum(r["normalized_score"] for r in successful_results)
    avg_normalized_score = total_normalized_score / len(successful_results)

    total_rescued = sum(r["total_rescued"] for r in successful_results)
    avg_total_rescued = total_rescued / len(successful_results)

    total_steps = sum(r["total_steps"] for r in successful_results)
    avg_total_steps = total_steps / len(successful_results)

    all_rescued_count = sum(1 for r in successful_results if r["all_rescued"])

    # Combined score is the average normalized score
    combined_score = avg_normalized_score

    public_metrics = {
        "num_scenarios": len(results),
        "num_successful": len(successful_results),
        "avg_total_rescued": f"{avg_total_rescued:.2f}",
        "avg_total_steps": f"{avg_total_steps:.2f}",
        "all_rescued_rate": f"{all_rescued_count}/{len(successful_results)}",
    }

    private_metrics = {
        "avg_normalized_score": avg_normalized_score,
        "min_normalized_score": min(r["normalized_score"] for r in successful_results),
        "max_normalized_score": max(r["normalized_score"] for r in successful_results),
        "scenario_scores": [
            {
                "scenario_idx": i,
                "normalized_score": r["normalized_score"],
                "total_rescued": r["total_rescued"],
                "total_steps": r["total_steps"],
            }
            for i, r in enumerate(results)
            if r.get("success", False)
        ],
    }

    metrics = {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
    }

    # Save detailed results for each scenario
    scenarios_file = os.path.join(results_dir, "scenarios_detail.json")
    try:
        scenarios_data = []
        for i, result in enumerate(results):
            if result.get("success", False):
                scenarios_data.append(
                    {
                        "scenario_idx": i,
                        "scenario_type": result.get("scenario_type", "unknown"),
                        "normalized_score": result["normalized_score"],
                        "total_rescued": result["total_rescued"],
                        "total_steps": result["total_steps"],
                        "all_rescued": result["all_rescued"],
                    }
                )

        with open(scenarios_file, "w") as f:
            json.dump(scenarios_data, f, indent=2)
        print(f"Scenario details saved to {scenarios_file}")
    except Exception as e:
        print(f"Error saving scenario details: {e}")
        metrics["scenarios_save_error"] = str(e)

    # Save rescue log from first scenario only
    if successful_results and "rescue_log" in successful_results[0]:
        log_file = os.path.join(results_dir, "rescue_log_scenario0.json")
        try:
            with open(log_file, "w") as f:
                json.dump(successful_results[0]["rescue_log"], f, indent=2)
            print(f"Rescue log (scenario 0) saved to {log_file}")
        except Exception as e:
            print(f"Error saving rescue log: {e}")
            metrics["log_save_error"] = str(e)

    return metrics


def main(
    program_path: str,
    results_dir: str,
    visualize: bool,
    num_scenarios: int = 7,
    grid_size: tuple[int, int] = (10, 10),
    num_targets: int = 15,
    num_vehicles: int = 5,
    num_obstacles: int = 10,
    add_target_rate: float = 0.0,
    add_obstacle_rate: float = 0.0,
    max_steps: int = 20,
):
    """Run the disaster rescue evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Running {num_scenarios} different scenarios")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    # Load the strategy function from the program
    spec = importlib.util.spec_from_file_location("rescue_module", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    strategy_fn = module.run_rescue_simulation

    # Run experiments
    all_results = []
    for i in range(num_scenarios):
        scenario_type, random_seed = get_scenario_settings(i)
        print(f"\nScenario {i+1}/{num_scenarios}: {scenario_type} (seed={random_seed})")

        simulator = RescueSimulator(
            grid_size=grid_size,
            num_targets=num_targets,
            num_vehicles=num_vehicles,
            num_obstacles=num_obstacles,
            random_seed=random_seed,
            scenario_type=scenario_type,
            add_target_rate=add_target_rate,
            add_obstacle_rate=add_obstacle_rate,
        )

        result = simulator.run_simulation(
            strategy_fn,
            max_steps=max_steps,
            save_visualization=visualize,
            results_dir=results_dir if visualize else None,
        )

        # Add scenario metadata to result
        result["scenario_type"] = scenario_type
        result["scenario_idx"] = i

        all_results.append(result)

        # Validate result
        is_valid, error_msg = validate_rescue_result(result)
        if is_valid:
            print(
                f"  ✓ Score: {result['normalized_score']:.4f}, "
                f"Rescued: {result['total_rescued']}/{result['total_targets']}, "
                f"Steps: {result['total_steps']}"
            )
        else:
            print(f"  ✗ Validation failed: {error_msg}")

    # Aggregate metrics
    metrics = aggregate_rescue_metrics(all_results, results_dir)

    # Check if all validations passed
    correct = all(
        validate_rescue_result(r)[0] for r in all_results if r.get("success", False)
    )
    error_msg = None if correct else "Some validations failed"

    # Save results
    save_json_results(results_dir, metrics, correct, error_msg)

    print("\n" + "=" * 60)
    if correct:
        print("✓ Evaluation and validation completed successfully.")
    else:
        print(f"✗ Evaluation or validation failed: {error_msg}")

    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disaster Rescue evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_rescue_simulation')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Create visualization GIF of the simulation (default: False)",
    )
    parsed_args = parser.parse_args()
    main(
        parsed_args.program_path,
        parsed_args.results_dir,
        parsed_args.visualize,
    )
