import pandas as pd
import os
from tqdm import tqdm
import torch
import gc
from instance_runner import ExperimentRunner
from solver import InstanceType
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--distancetype', type=str, default='cuda',
                    help='distancetype')
args = parser.parse_args()
distance_type = args.distancetype

# --- CONFIGURATION ---
TOPK_VALUES = [20, 40, 60, 80, 100]
INSTANCE_TYPES = [InstanceType.ATT, InstanceType.EUC_2D, InstanceType.GEO]
NUM_INSTANCES_PER_TYPE = 10
OUTPUT_FILE = "run_2.csv"

if(distance_type == 'GEO'):
    distance_type = InstanceType.GEO
elif(distance_type == 'EUC_2D'):
    distance_type = InstanceType.EUC_2D
elif(distance_type == 'ATT'):
    distance_type = InstanceType.ATT

if __name__ == "__main__":
    runner = ExperimentRunner(device='cuda')
    runner.add_json_instances('data/new_instances/', distance_type)
    #runner.add_instances_by_ids([x for x in range(100)], distance_type)
    runner.run(file=OUTPUT_FILE, distance_type=distance_type)
