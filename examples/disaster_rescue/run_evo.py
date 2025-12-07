#!/usr/bin/env python3
"""
Run evolution for the Disaster Rescue Task

This script configures and runs the evolution process to improve
rescue vehicle coordination strategies.
"""

from shinka.core import EvolutionConfig, EvolutionRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Configure job execution
job_config = LocalJobConfig(eval_program_path="evaluate.py")

# Configure parent selection strategy
strategy = "weighted"
if strategy == "uniform":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "hill_climbing":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=100.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )

# Configure database and island model
db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    # Inspiration parameters
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration parameters
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    **parent_config,
)

# System message describing the task for the LLM
search_task_sys_msg = """You are an expert in disaster response optimization and multi-agent coordination systems.

Task: Develop an effective strategy for coordinating multiple rescue vehicles to save disaster victims as quickly as possible, prioritizing high-severity cases.

Scenario:
- 30x30 grid with obstacles
- 15 targets with severity levels (1-10, higher = more urgent)
- 5 rescue vehicles
- targets and obstacles may appear dynamically over time
- Vehicles must navigate around obstacles to reach targets

Key Optimization Directions:
1. Prioritize high-severity targets early in the rescue operation
2. Minimize travel time by considering proximity and avoiding redundant assignments
3. Coordinate between vehicles to avoid multiple vehicles targeting the same victim
4. Balance between greedy local optimization and global strategy
5. Consider obstacle avoidance and path length in decision making
6. Implement lookahead to anticipate future assignments
7. Use clustering or zone assignment strategies for efficiency
8. Consider dynamic reassignment when new information becomes available

Scoring:
- Higher scores for rescuing high-severity targets quickly
- Penalty for delays (score decreases with time)
- Goal: Maximize weighted score = - Σ(severity * log(time_penalty))

The function should return the target ID that this vehicle should pursue.
If the function returns None, the vehicle will remain idle for this turn.
Be creative and develop sophisticated coordination strategies that outperform simple greedy approaches."""

# Configure evolution parameters
evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=100,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gemini-2.5-pro"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gemini-2.5-pro"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.py",
    results_dir="results_disaster_rescue",
)


def main():
    """Run the evolution experiment."""
    print("=" * 60)
    print("Disaster Rescue Task - Evolution Experiment")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Generations: {evo_config.num_generations}")
    print(f"  - Islands: {db_config.num_islands}")
    print(f"  - Parallel jobs: {evo_config.max_parallel_jobs}")
    print(f"  - Strategy: {strategy}")
    print("=" * 60)

    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()

    print("\n" + "=" * 60)
    print("Evolution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
