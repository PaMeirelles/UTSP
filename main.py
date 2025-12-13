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

# --- CONFIGURATION ---
TOPK_VALUES = [1, .5, .2, .1]
INSTANCE_TYPES = [
    InstanceType.ATT,
    InstanceType.EUC_2D,
    InstanceType.GEO
]

# --- 5-HOUR RUN SETTINGS (Target ~6s Avg) ---

# ILS: Cut iterations significantly to tame the large instances
# Previous Size 100 avg: 63s (50 iters). New target: ~12s -> 10 iters.
ILS_CONFIG = {
    "max_iter": 10,             # Reduced from 50 to 10
    "perturbation_strength": 3,
    "improvement_mode": "first",
    "temp_factor": 0.05,
    "min_temp_ratio": 1e-8,
    "cooling_rate": 0.95
}

# SA: Slightly faster cooling to match ILS average
# Previous ~9s. New target ~5-6s.
SA_CONFIG = {
    "temp_factor": 0.5,
    "min_temp_ratio": 1e-5,
    "cooling_rate": 0.99998     # Reduced from 0.99999
}

METHODS = [SolverMethod.SA, SolverMethod.ILS]
NUM_INSTANCES = 100
SIZES = [x for x in range(10, 101, 10)]
OUTPUT_FOLDER = "run_5h_final"
INSTANCE_FOLDER = "data/new_instances"
MAX_ID = 11109
MAX_WORKERS = 10


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
                    for method in METHODS:
                        for topk in TOPK_VALUES:
                            run_ids.append(RunID(instance_type, topk, pointer, method, instance_size))
                    
                    instances_added_per_size[instance_size] += 1
                    total_instances_added += 1
            
            pointer += 1
            if total_instances_added == NUM_INSTANCES * len(sizes):
                break
    return run_ids


def round_robin_sort(run_ids: List[RunID]) -> List[RunID]:
    """
    Sorts runs to distribute difficulty evenly.
    Iterates through instances, picking one from each size/type combo sequentially.
    """
    problems: Dict[Tuple[int, InstanceType, int], List[RunID]] = defaultdict(list)
    for r in run_ids:
        key = (r.size, r.instance_type, r.instance_id)
        problems[key].append(r)

    bucket: Dict[int, Dict[InstanceType, List[Tuple[int, InstanceType, int]]]] = defaultdict(lambda: defaultdict(list))
    for key in problems.keys():
        size, idx_type, _ = key
        bucket[size][idx_type].append(key)
        
    for s in bucket:
        for t in bucket[s]:
            bucket[s][t].sort(key=lambda x: x[2])

    sorted_runs = []
    max_count = 0
    for s in bucket:
        for t in bucket[s]:
            max_count = max(max_count, len(bucket[s][t]))
            
    sorted_sizes = sorted(SIZES)
    
    for k in range(max_count):
        for s in sorted_sizes:
            for t in INSTANCE_TYPES:
                if k < len(bucket[s][t]):
                    prob_key = bucket[s][t][k]
                    sorted_runs.extend(problems[prob_key])
                    
    return sorted_runs


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
    if calls:
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


def process_single_run(run_id: RunID, lock) -> Tuple[RunID, Optional[Dict], Optional[str]]:
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
            verbose=False,
            **solver_kwargs
        )

        if run_id.method == SolverMethod.ILS:
            save_stats(run_id, result.calls)

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
            "date_run": "13_12_25"
        }

        csv_file = os.path.join(OUTPUT_FOLDER, "summary_results.csv")
        fieldnames = list(csv_row.keys())

        with lock:
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)

        return run_id, csv_row, None

    except Exception as e:
        return run_id, None, str(e)


def main():
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Generating Run IDs...")
    run_ids = get_run_ids(SIZES)
    print(f"Total runs generated: {len(run_ids)}")

    run_ids = round_robin_sort(run_ids)
    print("Runs sorted (Round-Robin).")

    done_dict: Dict[RunID, bool] = {key: False for key in run_ids}
    fill_done_dict(done_dict)

    pending_runs = [rid for rid, done in done_dict.items() if not done]
    print(f"Runs pending: {len(pending_runs)}")

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
                    pbar.set_description(f"Run {run_id.instance_type.name} {run_id.instance_id}")
                    pbar.set_postfix({
                        "M": run_id.method.name,
                        "Sz": run_id.size,
                        "K": run_id.topk,
                        "Gap": f"{csv_data['gap']:.2%}"
                    })

            except Exception as e:
                tqdm.write(f"Critical Worker Error {run_id}: {e}")

            pbar.update(1)


if __name__ == "__main__":
    main()