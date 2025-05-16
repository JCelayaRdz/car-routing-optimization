import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import wilcoxon
import json

class ExperimentAnalyzer:
    def __init__(self, experiment_path: str):
        self.dataset_id = experiment_path
        with open(experiment_path, 'r') as f:
            self.experiment_data = json.load(f)

    def get_runs_by_config_id(self, config_id: int) -> List[dict]:
        """Return the list of runs for a given config_id."""
        for config in self.experiment_data:
            if config["config_id"] == config_id:
                return config["runs"]
        raise ValueError(f"No configuration with config_id={config_id} found.")

    def extract_metric_history(self, runs: List[dict], metric_key: str) -> List[List[float]]:
        return [run[metric_key] for run in runs if metric_key in run]

    def plot_metric_per_iteration(self, metric_history_list: List[List[float]], label: str, ylabel: str):
        for run_id, history in enumerate(metric_history_list):
            plt.plot(history, alpha=0.3, label=f"Run {run_id+1}" if run_id < 5 else None)
        avg = np.mean(metric_history_list, axis=0)
        plt.plot(avg, label="Average", color='black', linewidth=2)
        plt.title(f"{label} over iterations")
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.show()

    def compare_configs(self, config_ids: List[int], metric: str = "hv_history"):
        """Compare multiple configurations in terms of their metric evolution (e.g., HV)."""
        for config_id in config_ids:
            try:
                runs = self.get_runs_by_config_id(config_id)
                metric_histories = self.extract_metric_history(runs, metric)
                self.plot_metric_per_iteration(metric_histories, f"Config {config_id} - {metric}", ylabel=metric.upper())
            except ValueError as e:
                print(e)

    def summarize_config(self, config_id: int):
        runs = self.get_runs_by_config_id(config_id)
        runtimes = [run["runtime"] for run in runs]
        n_solutions = [run["n_solutions"] for run in runs]
        print(f"Summary for config_id={config_id}")
        print(f"Avg runtime: {np.mean(runtimes):.2f}s, Std: {np.std(runtimes):.2f}")
        print(f"Avg #solutions: {np.mean(n_solutions):.2f}, Std: {np.std(n_solutions):.2f}")

        if "hv_history" in runs[0]:
            final_hv = [run["hv_history"][-1] for run in runs]
            print(f"Avg Final HV: {np.mean(final_hv):.4f}, Std: {np.std(final_hv):.4f}")

        if "spread_history" in runs[0]:
            final_spread = [run["spread_history"][-1] for run in runs]
            print(f"Avg Final Spread: {np.mean(final_spread):.4f}, Std: {np.std(final_spread):.4f}")
    
    def plot_final_metric_boxplot(self, config_ids: list, metric: str = "hv_history", dataset=None):
        all_data = []
        labels = []
        for config_id in config_ids:
            try:
                runs = self.get_runs_by_config_id(config_id)
                if metric == "hv_history":
                    vals = [run[metric][-1] for run in runs if metric in run]
                else:
                    vals = [run[metric] for run in runs if metric in run]
                all_data.append(vals)
                labels.append(f"Config {config_id}")
            except Exception as e:
                print(e)
        plt.boxplot(all_data, labels=labels)
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.title(f"Final {metric.replace('_', ' ').upper()} by Config" + (f" (Dataset {dataset})" if dataset else ""))
        plt.grid(True)
        plt.show()

    def wilcoxon_configs(self, config_id_1, config_id_2, metric: str = "hv_history"):
        runs1 = self.get_runs_by_config_id(config_id_1)
        runs2 = self.get_runs_by_config_id(config_id_2)
        vals1 = [run[metric][-1] for run in runs1 if metric in run]
        vals2 = [run[metric][-1] for run in runs2 if metric in run]
        stat, p = wilcoxon(vals1, vals2)
        print(f"Wilcoxon test for {metric} between config {config_id_1} and {config_id_2}:")
        print(f"  Statistic={stat:.3f}, p-value={p:.5f}")
        if p < 0.05:
            print("  Result: Significant difference")
        else:
            print("  Result: No significant difference")