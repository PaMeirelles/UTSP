import os
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial import distance_matrix
import torch
from models import GNN
from utils import get_heat_map

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

    def _get_heatmap(self, device='cpu', temperature=3.5):
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
    # Test with a 100-node instance from test data
    test_data_path = './data/default_instances/test_tsp_instance_100.npy'

    if os.path.exists(test_data_path):
        print("Testing with 100-node instance from test data")
        tsp_instances = np.load(test_data_path)
        print(f"Loaded {tsp_instances.shape[0]} test instances with {tsp_instances.shape[1]} nodes each")

        # Create an instance from the first test case
        instance = Instance(
            instance_type=InstanceType.EUC_2D,
            instance_id=0,
            coordinates=tsp_instances[0]
        )
    else:
        # Fallback: try to load from new_instances folder
        print("Test data not found, trying to load from new_instances folder")
        instance = load_instance(0, InstanceType.EUC_2D)

    print(f"Instance has {instance.get_number_of_nodes()} nodes")
    print(f"Coordinates shape: {instance.coordinates.shape}")

    # Test heatmap generation
    print("\nGenerating heatmap...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        heatmap = instance._get_heatmap(device=device)
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Heatmap min: {heatmap.min():.6f}, max: {heatmap.max():.6f}")
        print(f"Heatmap mean: {heatmap.mean():.6f}")
        print("\nHeatmap generation successful!")
    except ValueError as e:
        print(f"Error: {e}")