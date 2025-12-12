import os
import csv
import json
import dataclasses
from typing import Dict, List
from pathlib import Path

from tqdm import tqdm

from heuristic.ils import NeighborhoodCall
from solver import InstanceType, SolverMethod, load_instance, SolverResult

# --- CONFIGURATION ---
TOPK_VALUES = [1, .5, .2, .1]
INSTANCE_TYPES = [InstanceType.ATT, InstanceType.EUC_2D, InstanceType.GEO]
METHODS = [SolverMethod.SA, SolverMethod.ILS]
NUM_INSTANCES = 1
SIZES = [x for x in range(10, 101, 10)]
OUTPUT_FOLDER = "run_12_12_25"
INSTANCE_FOLDER = "data/new_instances"
MAX_ID = 11109


@dataclasses.dataclass(frozen=True)  # Frozen allows it to be hashable for dict keys
class RunID:
    instance_type: InstanceType
    topk: float
    instance_id: int
    method: SolverMethod


def get_run_filename(run_id: RunID, extension: str) -> str:
    """Helper to generate consistent filenames."""
    # Example: run_ATT_SA_id10_topk0.1.json
    return f"{OUTPUT_FOLDER}/run_{run_id.instance_type.name}_{run_id.method.name}_id{run_id.instance_id}_topk{run_id.topk}.{extension}"


def get_run_ids(sizes: List[int]) -> List[RunID]:
    run_ids: List[RunID] = []

    for instance_type in INSTANCE_TYPES:
        instance_ids = []
        instances_added_per_size = {key: 0 for key in sizes}
        total_instances_added = 0
        pointer = 0

        while pointer < MAX_ID:
            try:
                instance = load_instance(pointer, instance_type)
            except Exception:
                # Skip if instance cannot be loaded
                pointer += 1
                continue

            instance_size = instance.get_number_of_nodes()

            if instance_size in instances_added_per_size:
                if instances_added_per_size[instance_size] < NUM_INSTANCES:
                    instance_ids.append(pointer)
                    total_instances_added += 1
                    instances_added_per_size[instance_size] += 1

            pointer += 1

            # Check if we have enough instances for this specific type
            if total_instances_added == NUM_INSTANCES * len(sizes):
                break

        # Generate combinations for this instance type
        for instance_id in instance_ids:
            for method in METHODS:
                for topk in TOPK_VALUES:
                    run_ids.append(RunID(instance_type, topk, instance_id, method))

    return run_ids


def fill_done_dict(done_dict: Dict[RunID, bool]) -> None:
    """
    Checks if the run is done.
    - For ILS: Checks if the specific JSON stats file exists.
    - For others (e.g., SA): Checks if the run exists in the summary CSV.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. Load completed runs from CSV (for non-ILS methods)
    csv_file = os.path.join(OUTPUT_FOLDER, "summary_results.csv")
    completed_csv_runs = set()

    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Construct a tuple signature for existing runs
                    # (instance_type_name, instance_id, method_name, topk)
                    run_sig = (
                        row['instance_type'],
                        int(row['instance_id']),
                        row['method'],
                        float(row['topk'])
                    )
                    completed_csv_runs.add(run_sig)
                except (ValueError, KeyError):
                    continue

    # 2. Update status for each run
    for run_id in done_dict.keys():
        if run_id.method == SolverMethod.ILS:
            # ILS saves a JSON file; existence of file = done
            json_path = get_run_filename(run_id, "json")
            if os.path.exists(json_path):
                done_dict[run_id] = True
        else:
            # Other methods only save to CSV; check existence in CSV set
            # We match using the same signature format as above
            run_sig = (
                run_id.instance_type.name,
                run_id.instance_id,
                run_id.method.name,
                run_id.topk
            )
            if run_sig in completed_csv_runs:
                done_dict[run_id] = True

def save_tabular_results(run_id: RunID, result: SolverResult, optimal_cost: float) -> None:
    csv_file = os.path.join(OUTPUT_FOLDER, "summary_results.csv")
    file_exists = os.path.isfile(csv_file)

    # Calculate gap
    gap = (result.cost - optimal_cost) / optimal_cost if optimal_cost != 0 else 0.0

    # Prepare data row
    row = {
        "instance_type": run_id.instance_type.name,
        "instance_id": run_id.instance_id,
        "method": run_id.method.name,
        "topk": run_id.topk,
        "optimal_cost": optimal_cost,
        "found_cost": result.cost,
        "gap": gap,
        "time_taken": result.time,
        "date_run": "12_12_25"
    }

    fieldnames = list(row.keys())

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_stats(run_id: RunID, calls: List[NeighborhoodCall]) -> None:
    file_path = get_run_filename(run_id, "json")

    calls_data = []
    for c in calls:
        # Extract the class name from the neighborhood instance (e.g., 'Shift', 'TwoOpt')
        # This prevents JSON serialization errors caused by the complex object
        nb_name = type(c.neighborhood_type).__name__

        calls_data.append({
            "duration": c.duration,
            "neighborhood": nb_name,
            "improvement": c.improvement
        })

    data = {
        "run_id": {
            "type": run_id.instance_type.name,
            "id": run_id.instance_id,
            "topk": run_id.topk
        },
        "neighborhood_calls": calls_data
    }

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def save_result(run_id: RunID, result: SolverResult, optimal_cost: float) -> None:
    save_tabular_results(run_id, result, optimal_cost)
    if run_id.method == SolverMethod.ILS:
        save_stats(run_id, result.calls)


def main():
    # Ensure output folder exists
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Generating Run IDs...")
    run_ids = get_run_ids(SIZES)
    print(f"Total runs generated: {len(run_ids)}")

    done_dict: Dict[RunID, bool] = {key: False for key in run_ids}
    fill_done_dict(done_dict)

    # Count pending runs
    pending_count = sum(1 for v in done_dict.values() if not v)
    print(f"Runs pending: {pending_count}")

    # Green progress bar configuration
    pbar = tqdm(
        done_dict.items(),
        total=len(done_dict),
        desc="Processing",
        unit="run",
        colour="green",
        dynamic_ncols=True
    )

    for key, status in pbar:
        if status:
            continue

        try:
            # Update description to current instance ID
            pbar.set_description(f"Inst: {key.instance_type.name} {key.instance_id}")

            # 1. Load & Solve
            instance = load_instance(key.instance_id, key.instance_type)
            optimal_cost = instance.calculate_tour_cost(instance.tour)

            size = instance.get_number_of_nodes()
            top_k_int = max(1, int(key.topk * size))

            result = instance.solve(
                method=key.method,
                topk=top_k_int,
                instance_type=key.instance_type,
                device='cuda'
            )

            save_result(key, result, optimal_cost)

            # 2. Calculate Gap
            gap = (result.cost - optimal_cost) / optimal_cost if optimal_cost != 0 else 0.0

            # 3. Update Postfix with Method, Size, TopK, and Gap
            pbar.set_postfix({
                "Method": key.method.name,
                "Sz": size,
                "K": key.topk,
                "Gap": f"{gap:.2%}"
            })

        except Exception as e:
            tqdm.write(f"Failed on run {key}: {e}")


if __name__ == "__main__":
    main()