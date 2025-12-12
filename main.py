import os
import csv
import json
import dataclasses
import random
import multiprocessing
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from heuristic.ils import NeighborhoodCall
from solver import InstanceType, SolverMethod, load_instance

random.seed(42)

# --- CONFIGURATION ---
TOPK_VALUES = [1, .5, .2, .1]
INSTANCE_TYPES = [
    InstanceType.ATT,
    InstanceType.EUC_2D,
    InstanceType.GEO
]

ILS_CONFIG = {
    "max_iter": 50,
    "perturbation_strength": 3,
    "improvement_mode": "first"
}

SA_CONFIG = {
    "initial_temp": 50000,
    "final_temp": 1,
    "cooling_rate": 1 - 1e-6
}

METHODS = [SolverMethod.SA, SolverMethod.ILS]
NUM_INSTANCES = 100
SIZES = [x for x in range(10, 101, 10)]
OUTPUT_FOLDER = "run_12_12_25"
INSTANCE_FOLDER = "data/new_instances"
MAX_ID = 11109
MAX_WORKERS =4


@dataclasses.dataclass(frozen=True)
class RunID:
    instance_type: InstanceType
    topk: float
    instance_id: int
    method: SolverMethod
    size: int

def get_run_filename(run_id: RunID, extension: str) -> str:
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
                pointer += 1
                continue
            instance_size = instance.get_number_of_nodes()
            if instance_size in instances_added_per_size:
                if instances_added_per_size[instance_size] < NUM_INSTANCES:
                    instance_ids.append((pointer, instance_size))
                    total_instances_added += 1
                    instances_added_per_size[instance_size] += 1
            pointer += 1
            if total_instances_added == NUM_INSTANCES * len(sizes):
                break
        for (inst_id, inst_size) in instance_ids:
            for method in METHODS:
                for topk in TOPK_VALUES:
                    run_ids.append(RunID(instance_type, topk, inst_id, method, inst_size))
    return run_ids


def stratified_sort(run_ids: List[RunID]) -> List[RunID]:
    groups = defaultdict(list)
    for r in run_ids:
        groups[r.size].append(r)
    rng = random.Random(42)
    sorted_sizes = sorted(groups.keys())
    for s in sorted_sizes:
        rng.shuffle(groups[s])
    interleaved = []
    max_len = max(len(g) for g in groups.values())
    for i in range(max_len):
        for s in sorted_sizes:
            if i < len(groups[s]):
                interleaved.append(groups[s][i])
    return interleaved


def fill_done_dict(done_dict: Dict[RunID, bool]) -> None:
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    csv_file = os.path.join(OUTPUT_FOLDER, "summary_results.csv")
    completed_csv_runs = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    run_sig = (
                        row['instance_type'],
                        int(row['instance_id']),
                        row['method'],
                        float(row['topk'])
                    )
                    completed_csv_runs.add(run_sig)
                except (ValueError, KeyError):
                    continue
    for run_id in done_dict.keys():
        if run_id.method == SolverMethod.ILS:
            json_path = get_run_filename(run_id, "json")
            if os.path.exists(json_path):
                done_dict[run_id] = True
        else:
            run_sig = (
                run_id.instance_type.name,
                run_id.instance_id,
                run_id.method.name,
                run_id.topk
            )
            if run_sig in completed_csv_runs:
                done_dict[run_id] = True


def save_stats(run_id: RunID, calls: List[NeighborhoodCall]) -> None:
    file_path = get_run_filename(run_id, "json")
    calls_data = []
    for c in calls:
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
            "topk": run_id.topk,
            "size": run_id.size
        },
        "neighborhood_calls": calls_data
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# --- WORKER FUNCTION ---
def process_single_run(run_id: RunID, lock) -> Tuple[RunID, Optional[Dict], Optional[str]]:
    """
    Worker now takes a lock and WRITES to CSV directly.
    """
    try:
        solver_kwargs = SA_CONFIG if run_id.method == SolverMethod.SA else ILS_CONFIG

        instance = load_instance(run_id.instance_id, run_id.instance_type)
        optimal_cost = instance.calculate_tour_cost(instance.tour)

        size = run_id.size
        top_k_int = max(1, int(run_id.topk * size))

        result = instance.solve(
            method=run_id.method,
            topk=top_k_int,
            instance_type=run_id.instance_type,
            device='cuda',
            **solver_kwargs
        )

        # 1. Save JSON Stats (Safe, unique file)
        if run_id.method == SolverMethod.ILS:
            save_stats(run_id, result.calls)

        # 2. Prepare Data
        gap = (result.cost - optimal_cost) / optimal_cost if optimal_cost != 0 else 0.0
        csv_row = {
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

        # 3. Write to CSV SAFELY using the Lock
        csv_file = os.path.join(OUTPUT_FOLDER, "summary_results.csv")
        fieldnames = list(csv_row.keys())

        with lock:
            # Check existence INSIDE lock to prevent double-header writing
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)

        # Return data just for the UI/Progress bar
        return run_id, csv_row, None

    except Exception as e:
        return run_id, None, str(e)


def main():
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Generating Run IDs...")
    run_ids = get_run_ids(SIZES)
    print(f"Total runs generated: {len(run_ids)}")

    run_ids = stratified_sort(run_ids)

    done_dict: Dict[RunID, bool] = {key: False for key in run_ids}
    fill_done_dict(done_dict)

    pending_runs = [rid for rid, done in done_dict.items() if not done]
    print(f"Runs pending: {len(pending_runs)}")

    # Create Manager and Lock
    manager = multiprocessing.Manager()
    file_lock = manager.Lock()

    pbar = tqdm(
        total=len(pending_runs),
        desc="Processing",
        unit="run",
        dynamic_ncols=True,
        colour="green",
        smoothing=0
    )

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass the lock to every worker
        future_to_run = {
            executor.submit(process_single_run, rid, file_lock): rid
            for rid in pending_runs
        }

        for future in as_completed(future_to_run):
            run_id = future_to_run[future]
            try:
                _, csv_data, error_msg = future.result()

                if error_msg:
                    tqdm.write(f"Failed on run {run_id}: {error_msg}")
                else:
                    # Update TQDM only (Writing is already done by worker)
                    pbar.set_description(f"Run {run_id.instance_type.name} {run_id.instance_id}")
                    pbar.set_postfix({
                        "Method": run_id.method.name,
                        "Sz": run_id.size,
                        "K": run_id.topk,
                        "Gap": f"{csv_data['gap']:.2%}"
                    })

            except Exception as e:
                tqdm.write(f"Critical Worker Error {run_id}: {e}")

            pbar.update(1)


if __name__ == "__main__":
    main()