import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import concurrent.futures
import time
import os

from solver import Instance, InstanceType, load_instance


# --- Import your existing classes here if in a different file ---
# from your_file import Instance, InstanceType, load_instance

@dataclass
class ExperimentMetrics:
    instance_name: str
    num_nodes: int
    solver_cost: float
    optimal_cost: Optional[float]
    gap_percent: Optional[float]
    total_time: float
    status: str  # 'SUCCESS', 'FAILED', 'NO_OPTIMAL'


class ExperimentRunner:
    def __init__(self,
                 device: str = 'cpu',
                 temperature: float = 3.5,
                 topk: int = 20,
                 timeout: int = 300,
                 verbose: bool = True):
        """
        Args:
            device: 'cpu' or 'cuda'
            temperature: Heatmap temperature
            topk: Top-k edges for MCTS
            timeout: Max seconds per instance
            verbose: Print progress
        """
        self.device = device
        self.temperature = temperature
        self.topk = topk
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[ExperimentMetrics] = []
        self.instances: List[Instance] = []

    def add_instance(self, instance: Instance):
        """Add a single instance object to the experiment queue."""
        self.instances.append(instance)

    def add_instances_by_ids(self, ids: List[int], instance_type: InstanceType):
        """Load and add multiple instances by ID."""
        for idx in ids:
            try:
                inst = load_instance(idx, instance_type)
                self.add_instance(inst)
            except FileNotFoundError:
                print(f"Warning: Instance {idx} of type {instance_type} not found.")

    def clear_instances(self):
        self.instances = []
        self.results = []

    def _calculate_gap(self, solver_cost: float, optimal_cost: float) -> float:
        if optimal_cost == 0:
            return 0.0
        return ((solver_cost - optimal_cost) / optimal_cost) * 100.0

    def _process_single_instance(self, instance: Instance) -> ExperimentMetrics:
        """
        Internal wrapper to solve one instance and calculate metrics.
        """
        start_total = time.time()

        # 1. Calculate Optimal Cost (Ground Truth) if available
        optimal_cost = None
        if instance.solution is not None:
            # We assume instance.solution is a list/array of indices.
            # We use the instance's own method to calculate cost to ensure consistency.
            # Note: We need to cast numpy array to list for the existing method
            try:
                # Assuming the Instance class has a method like this
                optimal_cost = instance._calculate_tour_cost(instance.solution.tolist())
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating optimal cost for {instance.get_name()}: {e}")

        # 2. Solve
        try:
            # result = instance.solve(...) is assumed to exist and return an object with a 'cost' and 'time' attribute
            result = instance.solve(
                device=self.device,
                temperature=self.temperature,
                topk=self.topk,
                timeout=self.timeout
            )
            # Assuming 'result' has a 'solve_time' attribute from the call to instance.solve
            solve_time = getattr(result, 'solve_time', time.time() - start_total)

            # 3. Calculate Gap
            gap = None
            status = 'SUCCESS'

            if optimal_cost is not None:
                gap = self._calculate_gap(result.cost, optimal_cost)
            else:
                status = 'NO_OPTIMAL'

            metrics = ExperimentMetrics(
                instance_name=instance.get_name(),
                num_nodes=instance.get_number_of_nodes(),
                solver_cost=result.cost,
                optimal_cost=optimal_cost,
                gap_percent=gap,
                total_time=time.time() - start_total,
                status=status
            )

        except Exception as e:
            print(f"Failed to solve {instance.get_name()}: {e}")
            metrics = ExperimentMetrics(
                instance_name=instance.get_name(),
                num_nodes=instance.get_number_of_nodes(),
                solver_cost=-1.0,
                optimal_cost=optimal_cost,
                gap_percent=None,
                total_time=time.time() - start_total,
                status=f'FAILED: {str(e)}'
            )

        return metrics

    ## New Methods Added Below

    # 1. Add a method to save
    def save_results(self, filename: str):
        """
        Saves the current experiment results to a file.
        The file format is inferred from the extension (e.g., .csv, .json).
        """
        df = self.get_summary()

        if df.empty:
            print("Warning: No results to save.")
            return

        try:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension == '.csv':
                df.to_csv(filename, index=False)
            elif file_extension == '.json':
                df.to_json(filename, orient='records', indent=4)
            elif file_extension in ['.xlsx', '.xls']:
                df.to_excel(filename, index=False)
            else:
                # Default to CSV if extension is unrecognized or missing
                df.to_csv(filename + '.csv', index=False)
                filename += '.csv'

            if self.verbose:
                print(f"✅ Successfully saved results to **{filename}**.")

        except Exception as e:
            print(f"🚨 Error saving results to {filename}: {e}")

    # 2 & 3. Add an optional file param in the run method and save at the end
    def run(self, max_workers: int = 1, file: Optional[str] = None) -> pd.DataFrame:
        """
        Run the experiments.

        Args:
            max_workers: Number of parallel processes.
                         WARNING: If using CUDA, keep max_workers=1 to avoid context errors.
                         If using CPU, you can increase this.
            file: Optional filename (e.g., 'results.csv') to save the summary to
                  after execution.
        """
        print(f"🚀 Starting experiment on {len(self.instances)} instances...")
        print(f"Config: Device={self.device}, Temp={self.temperature}, TopK={self.topk}")

        results_list = []

        if max_workers > 1:
            # Parallel Execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map instances to futures
                future_to_inst = {
                    executor.submit(self._process_single_instance, inst): inst
                    for inst in self.instances
                }

                # Iterate as they complete
                iterator = concurrent.futures.as_completed(future_to_inst)
                if self.verbose:
                    iterator = tqdm(iterator, total=len(self.instances), desc="Solving")

                for future in iterator:
                    # Append result; re-raise exception if one occurred
                    try:
                        results_list.append(future.result())
                    except Exception as exc:
                        print(f"Instance processing generated an exception: {exc}")
                        # You might want to skip or log this instance's result

        else:
            # Sequential Execution
            iterator = self.instances
            if self.verbose:
                iterator = tqdm(iterator, desc="Solving")

            for inst in iterator:
                results_list.append(self._process_single_instance(inst))

        self.results = results_list
        summary_df = self.get_summary()

        # Save results if a filename is provided
        if file is not None:
            self.save_results(file)

        return summary_df

    def get_summary(self) -> pd.DataFrame:
        """Convert results to a Pandas DataFrame."""
        df = pd.DataFrame([asdict(r) for r in self.results])
        return df

    def print_stats(self):
        """Print a quick statistical summary to console."""
        df = self.get_summary()
        if df.empty:
            print("No results to show.")
            return

        success_df = df[df['status'] == 'SUCCESS']

        print("\n=== Experiment Summary ===")
        print(f"Total Instances: {len(df)}")
        print(f"Successful Solves: {len(success_df)}")
        print(f"Failed: {len(df) - len(success_df)}")

        if not success_df.empty:
            print("\n--- Performance Metrics (Success only) ---")
            print(f"Avg Total Time: {success_df['total_time'].mean():.4f} s")
            # This line assumes 'solve_time' is now correctly added to ExperimentMetrics
            # or is otherwise available/calculated in _process_single_instance.
            # print(f"Avg Solver Time: {success_df['solve_time'].mean():.4f} s")

            if 'gap_percent' in success_df and success_df['gap_percent'].notna().any():
                print(f"Avg Gap: {success_df['gap_percent'].mean():.4f}%")
                print(f"Max Gap: {success_df['gap_percent'].max():.4f}%")
                print(f"Min Gap: {success_df['gap_percent'].min():.4f}%")
            else:
                print("Gap metrics unavailable (no optimal solutions found in data).")