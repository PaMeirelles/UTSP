import os
import random
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import json
from dataclasses import dataclass, field
from scipy.spatial import distance_matrix
import torch
import tempfile
import time
from models import GNN
from utils import get_heat_map
from cpp_interface import *
from heuristic.sa import SimulatedAnnealing
from heuristic.ils import IteratedLocalSearch, NeighborhoodCall
from heuristic.heuristic_tsp_solver import HeuristicTSPSolution

INSTANCE_FOLDER = 'data/new_instances'


class InstanceType(Enum):
    ATT = 0
    EUC_2D = 1
    GEO = 2


class SolverMethod(Enum):
    MCTS = "mcts"
    SA = "sa"
    ILS = "ils"


def instance_type_to_name(instance_type: InstanceType):
    if instance_type == InstanceType.ATT:
        return 'ATT'
    elif instance_type == InstanceType.EUC_2D:
        return 'EUC_2D'
    elif instance_type == InstanceType.GEO:
        return 'GEO'
    else:
        raise ValueError('Unknown instance type')


@dataclass
class SolverResult:
    time: float
    tour: List[int]
    cost: float
    calls: Optional[List[NeighborhoodCall]] = None


class Instance:
    def __init__(self, instance_type: InstanceType,
                 instance_id: int,
                 coordinates: List[Tuple[float, float]],
                 solution: Optional[List[float]] = None) -> None:
        self.instance_type = instance_type
        self.instance_id = instance_id
        self.coordinates = np.array(coordinates)
        if solution is not None:
            self.tour = np.array(solution)
        else:
            self.tour = None

    def get_name(self) -> str:
        return f"{instance_type_to_name(self.instance_type)}_{self.instance_id}"

    def get_number_of_nodes(self) -> int:
        return len(self.coordinates)

    def _add_dummies(self, target: int):
        current_n = self.get_number_of_nodes()
        missing = target - current_n

        if missing <= 0:
            return

        ref_node = self.coordinates[0]
        dummies = np.tile(ref_node, (missing, 1))
        self.coordinates = np.vstack((self.coordinates, dummies))

        if self.tour is not None:
            dummy_indices = np.arange(current_n, target)
            try:
                zero_pos = np.where(self.tour == 0)[0][0]
                self.tour = np.concatenate((
                    self.tour[:zero_pos + 1],
                    dummy_indices,
                    self.tour[zero_pos + 1:]
                ))
            except IndexError:
                self.tour = np.concatenate((self.tour, dummy_indices))

    def _get_heatmap(self, device='cpu', temperature=3.5) -> np.ndarray:
        sizes = [10, 50, 100]
        size = None
        for s in sizes:
            if self.get_number_of_nodes() <= s:
                size = s
                break
        if size is None:
            raise RuntimeError(f"Network size {size} is too big")
        self._add_dummies(size)
        num_nodes = self.get_number_of_nodes()

        model_configs = {
            10: {'hidden_dim': 64, 'nlayers': 2, 'rescale': 1.0},
            50: {'hidden_dim': 64, 'nlayers': 2, 'rescale': 1.0},
            100: {'hidden_dim': 64, 'nlayers': 2, 'rescale': 1.0},
        }

        if num_nodes not in model_configs:
            raise ValueError(f"No trained model available for size {num_nodes}")

        config = model_configs[num_nodes]
        hidden_dim = config['hidden_dim']
        nlayers = config['nlayers']
        rescale = config['rescale']

        distance_name = instance_type_to_name(self.instance_type)
        # Load the model
        model_path = f'Saved_Models/{distance_name}/TSP_{num_nodes}/scatgnn_layer_{nlayers}_hid_{hidden_dim}_model_210_temp_{temperature:.3f}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = GNN(input_dim=2, hidden_dim=hidden_dim, output_dim=num_nodes, n_layers=nlayers)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        coords = self.coordinates.copy()
        mean = np.mean(coords, axis=0)
        coords = coords - mean
        coords = rescale * coords

        dist_matrix = distance_matrix(coords, coords)

        coords_tensor = torch.FloatTensor(coords).unsqueeze(0).to(device)
        dist_tensor = torch.FloatTensor(dist_matrix).unsqueeze(0).to(device)

        adj = torch.exp(-1.0 * dist_tensor / temperature)
        mask = torch.ones(num_nodes, num_nodes).to(device)
        mask.fill_diagonal_(0)
        adj *= mask

        with torch.no_grad():
            output = model(coords_tensor, adj)
            heatmap = get_heat_map(output, num_nodes, device)

        return heatmap.squeeze(0).cpu().numpy()

    def _solve_instance(self, heatmap: np.ndarray,
                        method: SolverMethod = SolverMethod.MCTS,
                        topk: int = 20,
                        device: str = 'cpu',
                        timeout: int = 300,
                        **kwargs) -> SolverResult:
        if method == SolverMethod.MCTS:
            return self._solve_mcts(heatmap, topk, timeout)
        elif method in [SolverMethod.SA, SolverMethod.ILS]:
            return self._solve_python_heuristic(heatmap, method, topk, **kwargs)
        else:
            raise ValueError(f"Unknown solver method: {method}")

    def _solve_mcts(self, heatmap: np.ndarray, topk: int, timeout: int) -> SolverResult:
        num_nodes = self.get_number_of_nodes()
        if heatmap.shape != (num_nodes, num_nodes):
            raise ValueError(f"Heatmap shape mismatch")

        solver_params = get_solver_params(num_nodes)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            input_file = temp_dir_path / 'instance.txt'
            self._write_solver_input(input_file, heatmap, topk)

            solver_executable = ensure_solver_compiled(num_nodes)
            output_file = temp_dir_path / 'result.txt'

            start_time = time.time()
            run_solver(solver_executable, input_file, output_file, num_nodes,
                       solver_params, topk, timeout)
            solve_time = time.time() - start_time

            tour, cost = self._parse_solver_output(output_file)
            return SolverResult(time=solve_time, tour=tour, cost=cost)

    # --- Distance Calculation Logic ---

    def _to_geo_rad(self, coords: np.ndarray) -> np.ndarray:
        """Helper to convert TSPLIB GEO coordinates to radians."""
        deg = np.trunc(coords)
        mins = coords - deg
        # TSPLIB formula: PI * (deg + 5.0 * min / 3.0) / 180.0
        return np.pi * (deg + 5.0 * mins / 3.0) / 180.0

    def _calculate_distances(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """
        Calculates distances between two arrays of coordinates c1 and c2.
        Supports broadcasting (e.g., for matrix generation vs pairwise check).
        """
        if self.instance_type == InstanceType.EUC_2D:
            diff = c1 - c2
            return np.sqrt(np.sum(diff ** 2, axis=-1))

        elif self.instance_type == InstanceType.ATT:
            # TSPLIB Pseudo-Euclidean
            diff = c1 - c2
            sq_dist = np.sum(diff ** 2, axis=-1)
            # d_ij = ceil( sqrt( (dx^2 + dy^2) / 10 ) )
            return np.ceil(np.sqrt(sq_dist / 10.0))

        elif self.instance_type == InstanceType.GEO:
            # TSPLIB Geographical Distance
            r1 = self._to_geo_rad(c1)
            r2 = self._to_geo_rad(c2)

            lats1, lons1 = r1[..., 0], r1[..., 1]
            lats2, lons2 = r2[..., 0], r2[..., 1]

            RRR = 6378.388

            q1 = np.cos(lons1 - lons2)
            q2 = np.cos(lats1 - lats2)
            q3 = np.cos(lats1 + lats2)

            val = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
            val = np.clip(val, -1.0, 1.0)

            return np.floor(RRR * np.arccos(val) + 1.0)

        else:
            raise ValueError(f"Unsupported instance type: {self.instance_type}")

    def calculate_tour_cost(self, tour: List[int]) -> float:
        """
        Calculates the exact cost of the tour using the instance-specific metric.
        Vectorized for performance.
        """
        # Create coordinate arrays based on the tour
        # P: [node_0, node_1, ..., node_N]
        # P_next: [node_1, node_2, ..., node_0] (Shifted)

        tour_indices = np.array(tour)
        if len(tour_indices) < self.get_number_of_nodes():
            pass

        p = self.coordinates[tour_indices]
        p_next = np.roll(p, -1, axis=0)

        distances = self._calculate_distances(p, p_next)
        return float(np.sum(distances))

    def _solve_python_heuristic(self, heatmap: np.ndarray,
                                method: SolverMethod,
                                topk: int,
                                **kwargs) -> SolverResult:
        """
        Runs Python-based SA or ILS using unified distance logic.
        """
        start_time = time.time()

        # 1. Prepare Distance Matrix
        coords = self.coordinates
        dist_matrix_np = self._calculate_distances(coords[:, np.newaxis, :], coords[np.newaxis, :, :])
        dist_matrix = dist_matrix_np.tolist()
        heatmap_list = heatmap.tolist()

        # 2. Initialize Heuristic Solver & Construct Initial Solution
        solver = HeuristicTSPSolution(dist_matrix, heatmap_list, topk)
        solver.construct_solution()  # Cheapest Insertion

        # --- DYNAMIC TEMPERATURE SCALING ---
        # Calculate T0 based on the initial solution cost.
        # For SA: cost * 0.5 means we accept expensive moves early on.
        # For ILS: cost * 0.05 is usually enough for perturbations.
        initial_cost = solver.get_solution_cost()

        # Get factor from kwargs (defaulting to safe values if missing)
        temp_factor = kwargs.get('temp_factor', 0.1)

        # Calculate Start Temp
        initial_temp = initial_cost * temp_factor

        # Calculate End Temp (Scale-independent)
        # If min_temp_ratio is provided, use it. Otherwise default to a small fraction.
        min_temp_ratio = kwargs.get('min_temp_ratio', 1e-5)
        final_temp = initial_temp * min_temp_ratio

        final_solution = None
        calls = None

        # 3. Run Metaheuristic
        if method == SolverMethod.SA:
            # Cooling rate from kwargs
            cooling_rate = kwargs.get('cooling_rate', 0.9995)

            sa = SimulatedAnnealing(
                solution=solver,
                initial_temp=initial_temp,  # Dynamic
                final_temp=final_temp,  # Dynamic
                cooling_rate=cooling_rate
            )
            final_solution = sa.solve(verbose=kwargs.get('verbose', False))

        elif method == SolverMethod.ILS:
            max_iter = kwargs.get('max_iter', 100)
            perturbation_strength = kwargs.get('perturbation_strength', 3)
            improvement_mode = kwargs.get('improvement_mode', "first")
            cooling_rate = kwargs.get('cooling_rate', 0.95)

            ils = IteratedLocalSearch(
                solution=solver,
                max_iter=max_iter,
                perturbation_strength=perturbation_strength,
                initial_temp=initial_temp,  # Dynamic
                final_temp=final_temp,  # Dynamic (Used for cut-off check)
                cooling_rate=cooling_rate,
                improvement_mode=improvement_mode
            )
            final_solution, calls = ils.run(verbose=kwargs.get('verbose', False))

        solve_time = time.time() - start_time

        # 4. Extract Results
        tour = final_solution.tour
        if len(tour) > 0 and tour[0] == tour[-1] and len(tour) > 1:
            tour = tour[:-1]

        cost = self.calculate_tour_cost(tour)

        return SolverResult(time=solve_time, tour=tour, cost=cost, calls=calls)

    def _write_solver_input(self, filename: Path, heatmap: np.ndarray, topk: int):
        num_nodes = self.get_number_of_nodes()
        with open(filename, 'w') as f:
            coords_flat = self.coordinates.flatten()
            f.write(' '.join(map(str, coords_flat)))
            f.write('\n')
            f.write('output ')
            dummy_solution = list(range(1, num_nodes + 1)) + [1]
            f.write(' '.join(map(str, dummy_solution)))
            f.write('\n')
            top_indices = []
            top_values = []
            for i in range(num_nodes):
                node_edges = heatmap[i, :]
                node_edges = node_edges.copy()
                node_edges[i] = -1
                topk_idx = np.argsort(node_edges)[-topk:][::-1]
                topk_vals = node_edges[topk_idx]
                top_indices.extend((topk_idx + 1).tolist())
                top_values.extend(topk_vals.tolist())
            f.write('indices ')
            f.write(' '.join(map(str, top_indices)))
            f.write('\n')
            f.write('output ')
            f.write(' '.join(map(str, top_values)))
            f.write('\n')

    def _parse_solver_output(self, output_file: Path) -> Tuple[List[int], float]:
        if not output_file.exists():
            raise RuntimeError(f"Solver output file not found")
        with open(output_file, 'r') as f:
            lines = f.readlines()
        tour = None
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('Solution:'):
                tour_str = line.replace('Solution:', '').strip()
                tour = [int(x) - 1 for x in tour_str.split()]
                if len(tour) > 1 and tour[-1] == tour[0]:
                    tour = tour[:-1]
        if tour is None:
            raise RuntimeError("Could not parse tour")

        cost = self.calculate_tour_cost(tour)
        return tour, cost

    def solve(self,
              device: str = 'cpu',
              temperature: float = 3.5,
              topk: int = 20,
              method: SolverMethod = SolverMethod.MCTS,
              timeout: int = 300,
              **kwargs) -> SolverResult:
        heatmap = self._get_heatmap(device=device, temperature=temperature)

        result = self._solve_instance(
            heatmap,
            method=method,
            topk=topk,
            device=device,
            timeout=timeout,
            **kwargs
        )
        return result


def load_file(path: str) -> List[Instance]:
    filename = os.path.basename(path)
    type_name = os.path.splitext(filename)[0]

    type_map = {
        'ATT': InstanceType.ATT,
        'EUC_2D': InstanceType.EUC_2D,
        'GEO': InstanceType.GEO
    }

    if type_name not in type_map:
        raise ValueError(f"Unknown instance type: {type_name}")

    instance_type = type_map[type_name]

    with open(path, 'r') as f:
        data = json.load(f)

    instances = []
    for idx, item in enumerate(data):
        coords = item['coords']
        tour = item.get('tour', None)

        instance = Instance(
            instance_type=instance_type,
            instance_id=idx,
            coordinates=coords,
            solution=tour
        )
        instances.append(instance)

    return instances


def load_folder(path: str) -> List[Instance]:
    result = []
    folder_path = Path(path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {path}")

    json_files = list(folder_path.glob('*.json'))

    if not json_files:
        raise ValueError(f"No JSON files found in folder: {path}")

    for json_file in json_files:
        instances = load_file(str(json_file))
        if instances:
            result += instances

    return result


def save_instances(instances: List[Instance]) -> None:
    instances_by_type = {}
    for instance in instances:
        instance_type = instance.instance_type
        if instance_type not in instances_by_type:
            instances_by_type[instance_type] = []
        instances_by_type[instance_type].append(instance)

    base_path = Path(INSTANCE_FOLDER)
    base_path.mkdir(parents=True, exist_ok=True)

    for instance_type, type_instances in instances_by_type.items():
        type_name = instance_type_to_name(instance_type)
        type_folder = base_path / type_name
        type_folder.mkdir(exist_ok=True)

        for instance in type_instances:
            instance_name = instance.get_name()
            coords_file = type_folder / f"{instance_name}.npy"
            np.save(coords_file, instance.coordinates)

            if instance.tour is not None:
                sol_file = type_folder / f"{instance_name}_sol.npy"
                np.save(sol_file, instance.tour)
                print(f"Saved instance {instance_name}: {coords_file} and {sol_file}")
            else:
                print(f"Saved instance {instance_name}: {coords_file} (no solution)")

        print(f"Total: Saved {len(type_instances)} instances of type {type_name}")


def load_instance(instance_id: int, instance_type: InstanceType) -> Instance:
    type_name = instance_type_to_name(instance_type)
    instance_name = f"{type_name}_{instance_id}"

    base_path = Path(INSTANCE_FOLDER) / type_name
    coords_file = base_path / f"{instance_name}.npy"
    sol_file = base_path / f"{instance_name}_sol.npy"

    if not coords_file.exists():
        raise FileNotFoundError(f"Instance file not found: {coords_file}")

    coordinates = np.load(coords_file)

    solution = None
    if sol_file.exists():
        solution = np.load(sol_file)

    return Instance(
        instance_type=instance_type,
        instance_id=instance_id,
        coordinates=coordinates,
        solution=solution
    )


if __name__ == '__main__':
    random.seed(42)
    instance = load_instance(9999, InstanceType.ATT)
    print(instance.get_number_of_nodes())
    op_cost = instance.calculate_tour_cost(instance.tour)

    # Example debug run with dynamic factors
    start = time.time()
    res_sa = instance.solve(
        method=SolverMethod.ILS,
        device='cuda',
        verbose=True,
        topk=100,
        max_iter=100,
        perturbation_strength=3,
        temp_factor=0.05,  # Should calculate T0 based on cost
        min_temp_ratio=1e-8,  # Should calculate Tf based on T0
        cooling_rate=0.95
    )
    end = time.time()
    print(f"Total time: {end - start}")
    print(f"Result: {res_sa.cost:.2f}")