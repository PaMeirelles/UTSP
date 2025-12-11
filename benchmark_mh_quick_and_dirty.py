import pandas as pd
import numpy as np
import time
import torch
import warnings
from scipy.spatial import distance_matrix

# Import your existing modules
# Ensure your project structure allows these imports
from solver import load_instance, InstanceType, SolverMethod, Instance, SolverResult
from heuristic.heuristic_tsp_solver import HeuristicTSPSolution
from heuristic.ils import IteratedLocalSearch
from heuristic.sa import SimulatedAnnealing


# --- 1. Helper Function to Extract Stats from ILS ---
def extract_ils_stats(ils_solver):
    """
    Aggregates statistics from the raw list of NeighborhoodCalls.
    Returns a flat dictionary suitable for CSV rows.
    """
    # Temporary storage
    agg_stats = {}

    for call in ils_solver.calls:
        name = type(call.neighborhood_type).__name__

        if name not in agg_stats:
            agg_stats[name] = {
                'count': 0,
                'total_imp': 0.0,
                'total_time': 0.0
            }

        agg_stats[name]['count'] += 1
        agg_stats[name]['total_imp'] += call.improvement
        agg_stats[name]['total_time'] += call.duration

    # Flatten for CSV (e.g., 'Shift_calls', 'Shift_roi')
    flat_stats = {}
    all_neighborhoods = ['Shift', 'Switch', 'TwoOpt']  # Ensure we have columns even if 0 calls

    for name in all_neighborhoods:
        data = agg_stats.get(name, {'count': 0, 'total_imp': 0.0, 'total_time': 0.0})

        count = data['count']
        t_imp = data['total_imp']
        t_time = data['total_time']

        flat_stats[f"{name}_calls"] = count
        flat_stats[f"{name}_imp_tot"] = t_imp
        flat_stats[f"{name}_time_tot"] = t_time
        flat_stats[f"{name}_imp_avg"] = t_imp / count if count > 0 else 0.0
        flat_stats[f"{name}_time_avg"] = t_time / count if count > 0 else 0.0
        # ROI: Improvement per Second
        flat_stats[f"{name}_roi"] = t_imp / t_time if t_time > 0 else 0.0

    return flat_stats


# --- 2. Patch the Solver to Capture Stats ---
# We redefine the internal python solver method to extract stats before returning
def _solve_python_heuristic_patched(self, heatmap: np.ndarray,
                                    method: SolverMethod,
                                    topk: int,
                                    **kwargs) -> SolverResult:
    start_time = time.time()

    # A. Calculate Distance Matrix (Copied logic from your solver.py)
    if self.instance_type == InstanceType.EUC_2D:
        dist_matrix_np = distance_matrix(self.coordinates, self.coordinates)
    elif self.instance_type == InstanceType.ATT:
        diff = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        dist_matrix_np = np.ceil(np.sqrt(sq_dist / 10.0))
    elif self.instance_type == InstanceType.GEO:
        # Simplified GEO logic (assuming your solver.py logic is correct/imported)
        # For brevity, reusing the EUC logic or throwing error if you strictly need GEO in this script
        # Assuming we are running EUC_2D as requested.
        dist_matrix_np = distance_matrix(self.coordinates, self.coordinates)
    else:
        raise ValueError(f"Unsupported type: {self.instance_type}")

    dist_matrix = dist_matrix_np.tolist()
    heatmap_list = heatmap.tolist()

    # B. Initialize & Construct
    solver = HeuristicTSPSolution(dist_matrix, heatmap_list, topk)
    solver.construct_solution()

    final_solution = None
    captured_stats = {}

    # C. Run Method
    if method == SolverMethod.ILS:
        ils = IteratedLocalSearch(
            solution=solver,
            max_iter=kwargs.get('max_iter', 100),
            perturbation_strength=kwargs.get('perturbation_strength', 3),
            improvement_mode=kwargs.get('improvement_mode', "first")
        )
        final_solution = ils.run(report_stats=False)

        # [CRITICAL STEP] Extract the stats before 'ils' is lost
        captured_stats = extract_ils_stats(ils)

    elif method == SolverMethod.SA:
        # Fallback for SA if needed
        sa = SimulatedAnnealing(solution=solver, initial_temp=kwargs.get('initial_temp', 1000))
        final_solution = sa.solve()

    solve_time = time.time() - start_time

    # D. Finalize
    tour = final_solution.tour
    if len(tour) > 0 and tour[0] == tour[-1] and len(tour) > 1:
        tour = tour[:-1]

    cost = final_solution.get_solution_cost()

    # Return result AND attach stats dynamically
    res = SolverResult(time=solve_time, tour=tour, cost=cost)
    res.stats = captured_stats  # Attach stats dictionary to result object
    return res


# Apply the patch to the Instance class
Instance._solve_python_heuristic = _solve_python_heuristic_patched


# --- 3. Run Experiment ---
def run_experiment():
    # Config
    NUM_INSTANCES = 20
    TOP_K_VALUES = [20, 60, 100]
    MAX_ITER = 50
    PERTURBATION = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Starting Experiment on {DEVICE}...")

    all_records = []

    for topk in TOP_K_VALUES:
        print(f"\n--- Processing TopK = {topk} ---")

        for i in range(NUM_INSTANCES):
            try:
                # Load
                instance = load_instance(instance_id=i, instance_type=InstanceType.EUC_2D)

                # Solve (using patched method)
                result = instance.solve(
                    method=SolverMethod.ILS,
                    device=DEVICE,
                    topk=topk,
                    max_iter=MAX_ITER,
                    perturbation_strength=PERTURBATION,
                    verbose=False
                )

                # Record Data
                record = {
                    "Instance_ID": i,
                    "TopK": topk,
                    "Cost": result.cost,
                    "Time_Total": result.time,
                }

                # Merge the stats into the record
                if hasattr(result, 'stats'):
                    record.update(result.stats)

                all_records.append(record)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/100")

            except Exception as e:
                print(f"  Error instance {i}: {e}")

    # --- Save ---
    if all_records:
        df = pd.DataFrame(all_records)

        # Organize columns: ID, TopK, Cost, Time, then the stats
        cols = ['Instance_ID', 'TopK', 'Cost', 'Time_Total']
        stat_cols = [c for c in df.columns if c not in cols]
        df = df[cols + stat_cols]

        filename = "ils_detailed_stats.csv"
        df.to_csv(filename, index=False)
        print(f"\nSaved detailed statistics to {filename}")

        # Print a snippet of ROI stats
        roi_cols = [c for c in df.columns if 'roi' in c]
        print("\nAverage ROI (Improvement/Sec) by TopK:")
        print(df.groupby('TopK')[roi_cols].mean())
    else:
        print("No results collected.")


if __name__ == "__main__":
    run_experiment()