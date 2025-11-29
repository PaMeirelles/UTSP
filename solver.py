import os
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial import distance_matrix
import torch
import tempfile
import time
from models import GNN
from utils import get_heat_map
from cpp_interface import *

INSTANCE_FOLDER = 'data/new_instances'

class InstanceType(Enum):
    ATT = 0
    EUC_2D = 1
    GEO = 2

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


class Instance:
    def __init__(self, instance_type: InstanceType,
                 instance_id: int,
                 coordinates: List[Tuple[float, float]],
                 solution: Optional[List[float]] = None) -> None:
        self.instance_type = instance_type
        self.instance_id = instance_id
        self.coordinates = np.array(coordinates)
        if solution is not None:
            self.solution = np.array(solution)

    def get_name(self) -> str:
        return f"{instance_type_to_name(self.instance_type)}_{self.instance_id}"

    def get_number_of_nodes(self) -> int:
        return len(self.coordinates)

    def _get_heatmap(self, device='cpu', temperature=3.5) -> np.ndarray:
        """
        Uses one of the trained networks to obtain the heatmap.
        Returns an NxN numpy array with edge probabilities.

        Args:
            device: 'cuda' or 'cpu' for computation
            temperature: temperature parameter for adjacency matrix (default 3.5)

        Returns:
            np.ndarray: NxN heatmap matrix
        """
        num_nodes = self.get_number_of_nodes()

        # Map instance size to exact model configuration
        # Models are trained for specific sizes and must match exactly
        model_configs = {
            100: {'hidden_dim': 64, 'nlayers': 2, 'rescale': 1.0},
            200: {'hidden_dim': 64, 'nlayers': 2, 'rescale': 2.0},
            500: {'hidden_dim': 64, 'nlayers': 2, 'rescale': 4.0},
            1000: {'hidden_dim': 128, 'nlayers': 2, 'rescale': 4.0}
        }

        if num_nodes not in model_configs:
            available_sizes = list(model_configs.keys())
            raise ValueError(
                f"No trained model available for problem size {num_nodes}. "
                f"Available sizes: {available_sizes}"
            )

        model_size = num_nodes
        config = model_configs[num_nodes]
        hidden_dim = config['hidden_dim']
        nlayers = config['nlayers']
        rescale = config['rescale']

        # Load the model
        model_path = f'Saved_Models/TSP_{model_size}/scatgnn_layer_{nlayers}_hid_{hidden_dim}_model_210_temp_{temperature:.3f}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create model instance (output_dim must match training size, not actual instance size)
        model = GNN(input_dim=2, hidden_dim=hidden_dim, output_dim=model_size, n_layers=nlayers)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # Preprocess coordinates: normalize and rescale
        coords = self.coordinates.copy()
        mean = np.mean(coords, axis=0)
        coords = coords - mean
        coords = rescale * coords

        # Create adjacency matrix from distances
        dist_matrix = distance_matrix(coords, coords)

        # Convert to tensors and add batch dimension
        coords_tensor = torch.FloatTensor(coords).unsqueeze(0).to(device)  # [1, N, 2]
        dist_tensor = torch.FloatTensor(dist_matrix).unsqueeze(0).to(device)  # [1, N, N]

        # Create adjacency matrix with temperature
        adj = torch.exp(-1.0 * dist_tensor / temperature)

        # Mask diagonal
        mask = torch.ones(num_nodes, num_nodes).to(device)
        mask.fill_diagonal_(0)
        adj *= mask

        # Run inference
        with torch.no_grad():
            output = model(coords_tensor, adj)
            heatmap = get_heat_map(output, num_nodes, device)

        # Convert to numpy and remove batch dimension
        heatmap_np = heatmap.squeeze(0).cpu().numpy()

        return heatmap_np

    def _solve_instance(self, heatmap: np.ndarray, topk: int = 20,
                       device: str = 'cpu', timeout: int = 300) -> SolverResult:
        """
        Solves the TSP instance using the C++ MCTS solver guided by the heatmap.

        Args:
            heatmap: NxN numpy array with edge probabilities
            topk: Number of top edges to consider per node (default 20)
            device: Device for computation (not used in solver, kept for API consistency)
            timeout: Maximum time in seconds for the solver to run

        Returns:
            SolverResult with time, tour, and cost
        """
        num_nodes = self.get_number_of_nodes()

        # Validate heatmap
        if heatmap.shape != (num_nodes, num_nodes):
            raise ValueError(f"Heatmap shape {heatmap.shape} doesn't match number of nodes {num_nodes}")

        # Prepare solver parameters based on problem size
        solver_params = get_solver_params(num_nodes)

        # Create temporary directory for solver I/O
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Write input file
            input_file = temp_dir_path / 'instance.txt'
            self._write_solver_input(input_file, heatmap, topk)

            # Compile solver if needed
            solver_executable = ensure_solver_compiled(num_nodes)

            # Prepare output file
            output_file = temp_dir_path / 'result.txt'

            # Run solver
            start_time = time.time()
            run_solver(
                solver_executable,
                input_file,
                output_file,
                num_nodes,
                solver_params,
                topk,
                timeout
            )
            solve_time = time.time() - start_time

            # Parse results
            tour, cost = self._parse_solver_output(output_file)

            return SolverResult(time=solve_time, tour=tour, cost=cost)

    def _write_solver_input(self, filename: Path, heatmap: np.ndarray, topk: int):
        """Write input file in the format expected by the C++ solver."""
        num_nodes = self.get_number_of_nodes()

        with open(filename, 'w') as f:
            # Write coordinates: x1 y1 x2 y2 ... xn yn
            coords_flat = self.coordinates.flatten()
            f.write(' '.join(map(str, coords_flat)))
            f.write('\n')

            # Write "output" followed by a dummy solution (1 2 3 ... n 1)
            # The solver doesn't actually use this, but format requires it
            f.write('output ')
            dummy_solution = list(range(1, num_nodes + 1)) + [1]
            f.write(' '.join(map(str, dummy_solution)))
            f.write('\n')

            # Extract top-k edges for each node based on heatmap
            top_indices = []
            top_values = []

            for i in range(num_nodes):
                # Get top-k neighbors for node i
                node_edges = heatmap[i, :]
                # Exclude self-loops
                node_edges = node_edges.copy()
                node_edges[i] = -1

                # Get top-k indices (1-indexed for C++)
                topk_idx = np.argsort(node_edges)[-topk:][::-1]
                topk_vals = node_edges[topk_idx]

                top_indices.extend((topk_idx + 1).tolist())  # Convert to 1-indexed
                top_values.extend(topk_vals.tolist())

            # Write "indices" followed by top-k indices for each node
            f.write('indices ')
            f.write(' '.join(map(str, top_indices)))
            f.write('\n')

            # Write "output" followed by heatmap values for those edges
            f.write('output ')
            f.write(' '.join(map(str, top_values)))
            f.write('\n')

    def _parse_solver_output(self, output_file: Path) -> Tuple[List[int], float]:
        """Parse the solver output file to extract tour and cost."""
        if not output_file.exists():
            raise RuntimeError(f"Solver output file not found: {output_file}")

        with open(output_file, 'r') as f:
            lines = f.readlines()

        tour = None
        cost = None

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for cost in lines like "MCTS Distance:7.123456"
            if 'MCTS' in line or 'Distance' in line:
                # Extract cost from the line
                parts = line.split()
                for j, part in enumerate(parts):
                    if 'Distance' in part and j + 1 < len(parts):
                        try:
                            cost_str = parts[j + 1]
                            cost = float(cost_str)
                            break
                        except ValueError:
                            continue

            # Look for solution line
            if line.startswith('Solution:'):
                # Parse tour from "Solution: 1 2 3 ... n"
                tour_str = line.replace('Solution:', '').strip()
                tour = [int(x) - 1 for x in tour_str.split()]  # Convert to 0-indexed
                # Remove duplicate start city if present
                if len(tour) > 1 and tour[-1] == tour[0]:
                    tour = tour[:-1]

        if tour is None:
            raise RuntimeError("Could not parse tour from solver output")

        if cost is None:
            # Calculate cost if not found in output
            cost = self._calculate_tour_cost(tour)

        return tour, cost

    def _calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate the total cost of a tour."""
        cost = 0.0
        num_nodes = len(tour)

        for i in range(num_nodes):
            city1 = tour[i]
            city2 = tour[(i + 1) % num_nodes]

            coord1 = self.coordinates[city1]
            coord2 = self.coordinates[city2]

            # Euclidean distance
            dist = np.sqrt(np.sum((coord1 - coord2) ** 2))
            cost += dist

        return cost

    def solve(self, device: str = 'cpu', temperature: float = 3.5,
              topk: int = 20, timeout: int = 300) -> SolverResult:
        """
        Complete end-to-end solve: generate heatmap and solve TSP.

        Args:
            device: Device for heatmap generation ('cpu' or 'cuda')
            temperature: Temperature parameter for heatmap generation
            topk: Number of top edges to consider per node
            timeout: Maximum time in seconds for the solver to run

        Returns:
            SolverResult with time, tour, and cost
        """
        # Generate heatmap
        print(f"Generating heatmap for {self.get_number_of_nodes()}-node instance...")
        heatmap = self._get_heatmap(device=device, temperature=temperature)

        # Solve using heatmap
        print("Solving TSP instance...")
        result = self._solve_instance(heatmap, topk=topk, device=device, timeout=timeout)

        return result


def load_file(path: str) -> List[Instance]:
    # Each file is named <instance_type>.json
    # Inside, there is a list of jsons with fields "coords" and 'tour'

    # Extract instance type from filename
    filename = os.path.basename(path)
    type_name = os.path.splitext(filename)[0]

    # Map filename to InstanceType
    type_map = {
        'ATT': InstanceType.ATT,
        'EUC_2D': InstanceType.EUC_2D,
        'GEO': InstanceType.GEO
    }

    if type_name not in type_map:
        raise ValueError(f"Unknown instance type: {type_name}")

    instance_type = type_map[type_name]

    # Load JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    # Parse instances
    instances = []
    for idx, item in enumerate(data):
        coords = item['coords']
        tour = item.get('tour', None)  # Optional

        instance = Instance(
            instance_type=instance_type,
            instance_id=idx,
            coordinates=coords,
            solution=tour
        )
        instances.append(instance)

    return instances

def load_folder(path: str) -> List[Instance]:
    # same but for folders
    # one file per instance type

    result = []

    # Find all JSON files in the folder
    folder_path = Path(path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {path}")

    json_files = list(folder_path.glob('*.json'))

    if not json_files:
        raise ValueError(f"No JSON files found in folder: {path}")

    # Load each file
    for json_file in json_files:
        instances = load_file(str(json_file))
        if instances:
            instance_type = instances[0].instance_type
            result += instances

    return result

def save_instances(instances: List[Instance]) -> None:
    # save inside 'data/new_instances'
    # one subfolder per type

    # Group instances by type
    instances_by_type = {}
    for instance in instances:
        instance_type = instance.instance_type
        if instance_type not in instances_by_type:
            instances_by_type[instance_type] = []
        instances_by_type[instance_type].append(instance)

    # Create base directory
    base_path = Path(INSTANCE_FOLDER)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save each type to a separate subfolder
    for instance_type, type_instances in instances_by_type.items():
        # Create subfolder for this type
        type_name = instance_type_to_name(instance_type)
        type_folder = base_path / type_name
        type_folder.mkdir(exist_ok=True)

        # Save each instance separately
        for instance in type_instances:
            instance_name = instance.get_name()

            # Save coordinates as {name}.npy
            coords_file = type_folder / f"{instance_name}.npy"
            np.save(coords_file, instance.coordinates)

            # Save solution as {name}_sol.npy if it exists
            if instance.solution is not None:
                sol_file = type_folder / f"{instance_name}_sol.npy"
                np.save(sol_file, instance.solution)
                print(f"Saved instance {instance_name}: {coords_file} and {sol_file}")
            else:
                print(f"Saved instance {instance_name}: {coords_file} (no solution)")

        print(f"Total: Saved {len(type_instances)} instances of type {type_name}")

def load_instance(instance_id: int, instance_type: InstanceType) -> Instance:
    # Load a specific instance from the folder
    type_name = instance_type_to_name(instance_type)
    instance_name = f"{type_name}_{instance_id}"

    # Construct file paths
    base_path = Path(INSTANCE_FOLDER) / type_name
    coords_file = base_path / f"{instance_name}.npy"
    sol_file = base_path / f"{instance_name}_sol.npy"

    # Check if coordinates file exists
    if not coords_file.exists():
        raise FileNotFoundError(f"Instance file not found: {coords_file}")

    # Load coordinates
    coordinates = np.load(coords_file)

    # Load solution if it exists
    solution = None
    if sol_file.exists():
        solution = np.load(sol_file)

    # Create and return instance
    return Instance(
        instance_type=instance_type,
        instance_id=instance_id,
        coordinates=coordinates,
        solution=solution
    )

if __name__ == '__main__':
    # instance = load_instance(0, InstanceType.ATT)
    np.random.seed(42)
    coordinates = list(np.random.rand(100, 2))

    instance = Instance(
        instance_type=InstanceType.EUC_2D,
        instance_id=0,
        coordinates=coordinates
    )
    instance.solve(device='cuda')