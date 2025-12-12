from typing import Dict, List
from heuristic.ils import NeighborhoodCall
from solver import InstanceType, SolverMethod, instance_type_to_name, load_instance, SolverResult
from dataclasses import dataclass

# --- CONFIGURATION ---
TOPK_VALUES = [1, .5, .2, .1]
INSTANCE_TYPES = [InstanceType.ATT, InstanceType.EUC_2D, InstanceType.GEO]
METHODS = [SolverMethod.SA, SolverMethod.ILS]
NUM_INSTANCES = 100
SIZES = [x for x in range(10, 101, 10)]
OUTPUT_FOLDER = "run_12_12_25"
INSTANCE_FOLDER = "data/new_instances"
MAX_ID = 11109

@dataclass
class RunID:
    instance_type: InstanceType
    topk: float
    instance_id: int
    method: SolverMethod

def get_run_ids(sizes: List[int]) -> List[RunID]:
    run_ids:List[RunID] = []
    for instance_type in INSTANCE_TYPES:
        instance_ids = []
        instances_added_per_size = {key: 0 for key in sizes}
        total_instances_added = 0
        pointer = 0
        while pointer < MAX_ID:
            instance = load_instance(pointer, instance_type)
            instance_size = instance.get_number_of_nodes()
            if instances_added_per_size[instance_size] < NUM_INSTANCES:
                instance_ids.append(pointer)
                total_instances_added += 1
                instances_added_per_size[instance_size] += 1
            else:
                continue
            pointer += 1
            if total_instances_added == NUM_INSTANCES * len(sizes):
                break

        for method in METHODS:
            for topk in TOPK_VALUES:
                for instance_id in instance_ids:
                    run_ids.append(RunID(instance_type, topk, instance_id, method))

        return run_ids

def fill_done_dict(done_dict: Dict[RunID, bool]) -> None:
    # Goes over a done dict and marks as 'True' when the record exists and 'False' otherwise
    pass

def save_tabular_results(run_id: RunID, result: SolverResult, optimal_cost: float) -> None:
    found_cost = result.cost

    # Saves all components of runID, the time taken and the gap from optimal
    # csv format
    pass

def save_stats(run_id: RunID, calls: List[NeighborhoodCall]) -> None:
    # Saves, in json format, the calls by runID
    pass

def save_result(run_id: RunID, result: SolverResult, optimal_cost: float) -> None:
    save_tabular_results(run_id, result, optimal_cost)
    save_stats(run_id, result.calls)
    pass

def main():
    run_ids = get_run_ids(SIZES)
    done_dict: Dict[RunID, bool] = {key: False for key in run_ids}
    fill_done_dict(done_dict)
    for key, status in done_dict.items():
        if status: continue

        instance = load_instance(key.instance_id, key.instance_type)

        optimal_tour = instance.tour
        optimal_cost = optimal_tour.get_cost()

        size = instance.get_number_of_nodes()
        top_k: int = int(key.topk * size)

        result = instance.solve(
            method=key.method,
            topk=top_k,
            instance_type=key.instance_type,
            device='cuda'
        )

main()