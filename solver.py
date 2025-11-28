from enum import Enum
from typing import Optional, List, Tuple, Dict
import numpy as np
import json
import os
from pathlib import Path

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


class Instance:
    def __init__(self, instance_type: InstanceType,
                 instance_id: int,
                 coordinates: List[Tuple[float, float]],
                 solution: Optional[List[float]] = None) -> None:
        self.instance_type = instance_type
        self.instance_id = instance_id
        self.coordinates = np.array(coordinates)
        if solution:
            self.solution = np.array(solution)

    def get_name(self) -> str:
        return f"{instance_type_to_name(self.instance_type)}_{self.instance_id}"


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



if __name__ == '__main__':
    instances = load_folder("data/raw_json")
    save_instances(instances,)