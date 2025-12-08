import pandas as pd
from tqdm import tqdm
import time
from typing import List
from solver import load_instance, InstanceType, SolverMethod

# --- CONFIGURATION ---
# Define the range of Top-K values you want to test
TOPK_VALUES = [20, 40, 60, 80, 100]

# Define the methods to run
METHODS = [SolverMethod.SA, SolverMethod.ILS]

# Instance settings
INSTANCE_TYPES = [InstanceType.EUC_2D, InstanceType.GEO, InstanceType.ATT]
INSTANCE_IDS = range(10)  # Run on the first 20 instances (adjust as needed)
DEVICE = 'cuda'  # Use 'cpu' if CUDA is not available

# Hyperparameters for Heuristics (passed as kwargs to solve)
# These override the defaults in solver.py
SA_PARAMS = {
    'initial_temp': 2000,
    'cooling_rate': 0.9995,
    'verbose': False
}

ILS_PARAMS = {
    'max_iter': 100,
    'perturbation_strength': 3,
    'verbose': False
}


def run_benchmark(instance_type, instance_ids: List[int], output_file: str = "heuristic_results.csv"):
    results = []

    print(f"--- Starting Benchmark ---")
    print(f"Methods: {[m.value for m in METHODS]}")
    print(f"Top-K: {TOPK_VALUES}")
    print(f"Instances: {len(instance_ids)} ({instance_type.name})")

    # Load instances first to avoid reloading them from disk multiple times
    instances = []
    print("\nLoading instances...")
    for idx in tqdm(instance_ids, desc="Loading"):
        try:
            inst = load_instance(idx, instance_type)
            instances.append(inst)
        except Exception as e:
            print(f"Warning: Could not load instance {idx}: {e}")

    # Main Benchmark Loop
    # We iterate: Instance -> TopK -> Method
    total_iterations = len(instances) * len(TOPK_VALUES) * len(METHODS)
    pbar = tqdm(total=total_iterations, desc="Solving")

    for inst in instances:
        for topk in TOPK_VALUES:
            for method in METHODS:

                # Select parameters based on method
                kwargs = {}
                if method == SolverMethod.SA:
                    kwargs = SA_PARAMS
                elif method == SolverMethod.ILS:
                    kwargs = ILS_PARAMS

                try:
                    start_time = time.time()

                    # Run the solver
                    res = inst.solve(
                        device=DEVICE,
                        temperature=3.5,  # Temperature for Heatmap generation
                        topk=topk,
                        method=method,
                        timeout=300,
                        **kwargs  # Pass specific params (e.g. max_iter)
                    )

                    elapsed = time.time() - start_time

                    # Store results
                    results.append({
                        "instance_id": inst.instance_id,
                        "num_nodes": inst.get_number_of_nodes(),
                        "method": method.value,
                        "top_k": topk,
                        "cost": res.cost,
                        "time": res.time,
                        "total_time": elapsed,  # Includes heatmap generation
                        "tour_length": len(res.tour) if res.tour else 0
                    })

                except Exception as e:
                    print(f"\nError solving Instance {inst.instance_id} [{method.value}, k={topk}]: {e}")
                    # Optionally append a failure record
                    results.append({
                        "instance_id": inst.instance_id,
                        "num_nodes": inst.get_number_of_nodes(),
                        "method": method.value,
                        "top_k": topk,
                        "cost": -1,
                        "time": 0,
                        "total_time": 0,
                        "tour_length": 0
                    })

                pbar.update(1)

    pbar.close()

    # Create DataFrame and Save
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")

        # Display a quick summary
        print("\n--- Summary (Average Cost) ---")
        summary = df[df['cost'] > 0].groupby(['method', 'top_k'])['cost'].mean()
        print(summary)
    else:
        print("\n❌ No results generated.")


if __name__ == "__main__":
    for instance_type in INSTANCE_TYPES:
        run_benchmark(instance_type, list(INSTANCE_IDS))