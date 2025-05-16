from dataclasses import dataclass
from typing import List
import json
import os
import time
from pathlib import Path
from metaheuristics.nsga.nsga_graph import NSGAGraphOptimizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

@dataclass
class NSGAExperimentHyperparams:
    config_id: int
    population_size: int
    generations: int
    penalty_signal: float
    penalty_lanes_factor: float

class NSGAExperimentExecuter:
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            dataset_list = json.load(f)
            self.datasets = {str(d["id"]): d for d in dataset_list}
        self.optimizer = None

    def run_experiment(self, dataset_id: str, hyperparams: NSGAExperimentHyperparams, n_repeat: int = 31):
        dataset = self.datasets[dataset_id]
        output_path = f"experiments/nsga/experiment_{dataset_id}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                experiment_results = json.load(f)
        else:
            experiment_results = []

        config_entry = {
            "config_id": hyperparams.config_id,
            "params": {
                "population_size": hyperparams.population_size,
                "generations": hyperparams.generations,
                "penalty_signal": hyperparams.penalty_signal,
                "penalty_lanes_factor": hyperparams.penalty_lanes_factor
            },
            "runs": []
        }

        for i in range(n_repeat):
            print(f"Run {i + 1}/{n_repeat} for dataset {dataset_id} with config {hyperparams.config_id}")

            self.optimizer = NSGAGraphOptimizer(
                graph_edges_path=Path("data/edges_clean.json"),
                graph_nodes_path=Path("data/nodes_clean.json"),
                origin=int(dataset["origin_node"]["osmid"]),
                destination=int(dataset["destination_node"]["osmid"]),
                vehicle_allowed_in_lez=bool(dataset["vehicle_allowed_in_lez"]),
                population_size=hyperparams.population_size,
                generations=hyperparams.generations,
                max_archive_size=25,
                penalty_signal=hyperparams.penalty_signal,
                penalty_lanes_factor=hyperparams.penalty_lanes_factor
            )

            start_time = time.time()
            self.optimizer.run()
            end_time = time.time()
            runtime = end_time - start_time

            archive = self.optimizer.archive[:25]
            pareto_data = [
                {
                    "solution": [(u, v, k) for (u, v, k) in route],
                    "objectives": list(fitness)
                }
                for (route, fitness) in archive
            ]

            config_entry["runs"].append({
                "run": i,
                "n_evaluations": self.optimizer.n_evaluations,
                "runtime": runtime,
                "n_solutions": len(pareto_data),
                "pareto_front": pareto_data,
                "hv_history": self.optimizer.history_hv,
                "spread_history": self.optimizer.history_spread,
            })

        experiment_results.append(config_entry)

        with open(output_path, 'w') as f:
            json.dump(experiment_results, f, indent=4)

    @property
    def get_optimizer(self) -> NSGAGraphOptimizer:
        return self.optimizer