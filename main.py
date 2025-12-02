import pandas as pd
import os
from tqdm import tqdm
import torch
import gc
from instance_runner import ExperimentRunner
from solver import InstanceType

# --- CONFIGURATION ---
TOPK_VALUES = [20, 40, 60, 80, 100]
INSTANCE_TYPES = [InstanceType.ATT, InstanceType.EUC_2D, InstanceType.GEO]
NUM_INSTANCES_PER_TYPE = 10
OUTPUT_FILE = "run_1.csv"

if __name__ == "__main__":
    runner = ExperimentRunner(device='cuda')
    runner.add_instances_by_ids([x for x in range(100)], InstanceType.EUC_2D)
    runner.run(file=OUTPUT_FILE)
