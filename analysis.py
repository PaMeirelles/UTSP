import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import os
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. SETUP & STYLE
# ==========================================

plt.style.use('default')
sns.set_context("notebook", font_scale=1.1)
sns.set_style("whitegrid", {'grid.color': '.9'})


def set_plot_header(ax, title, y_label=None, x_label=None):
    ax.set_title(title, fontsize=14, color='#003366', fontweight='bold', pad=15)
    if y_label: ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    if x_label: ax.set_xlabel(x_label, fontsize=11, fontweight='bold')


# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================

print("Step 1/5: Loading summary results for optimal costs...")
summary_file = 'run_fast/summary_results.csv'
optimals_map = {}

# Load benchmarks for ROI calculation
if os.path.exists(summary_file):
    try:
        summ_df = pd.read_csv(summary_file)
        # Create lookup: (instance_type, instance_id) -> optimal_cost
        summ_df = summ_df.drop_duplicates(subset=['instance_type', 'instance_id'])
        optimals_map = summ_df.set_index(['instance_type', 'instance_id'])['optimal_cost'].to_dict()
        print(f"   Loaded {len(optimals_map)} optimal cost entries.")
    except Exception as e:
        print(f"   Warning: Could not read summary_results.csv ({e}). ROI will be approximate.")

print("Step 2/5: Finding JSON run files...")
# Adjust path as needed (e.g., "results/calls/*.json")
files = glob.glob("run_fast/*.json")
# Filter for likely run files
files = [f for f in files if f.endswith('.json') and ("run_" in f or "ILS_" in f)]
print(f"   Found {len(files)} files to process.")

data_rows = []
# Dictionary to store transition logic: transitions[type][topk]
transition_data = {}
unique_neighborhoods = set()

print("Step 3/5: Processing files...")
for file_path in tqdm(files, desc="Parsing JSONs"):
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)

        # --- Metadata Extraction ---
        if 'run_id' in content:
            meta = content['run_id']
            i_type = meta.get('type', 'Unknown')
            i_id = meta.get('id', 0)
            topk = meta.get('topk', 0)
            calls = content.get('neighborhood_calls', [])
        else:
            continue

        # Get optimal cost for this instance (default to 1 to avoid division by zero)
        optimal_obj = optimals_map.get((i_type, i_id), 1)

        # --- Heatmap Logic Prep ---
        if i_type not in transition_data: transition_data[i_type] = {}
        if topk not in transition_data[i_type]:
            transition_data[i_type][topk] = {'success': {}, 'attempts': {}}
        t_store = transition_data[i_type][topk]

        failed_in_cycle = set()

        for step in calls:
            neigh = step.get('neighborhood')
            # Normalize names
            if neigh == 'Swap': neigh = 'Migrate'
            # REMOVED: elif neigh == 'Switch': neigh = 'Flip'  <-- User requested to keep 'Switch'

            unique_neighborhoods.add(neigh)

            # Improvement is typically negative (cost reduction)
            improvement = step.get('improvement', 0)
            duration = step.get('duration', 0)

            # --- Store Row Data ---
            data_rows.append({
                'instance_type': i_type,
                'instance_id': i_id,
                'topk': topk,
                'neighborhood': neigh,
                'improvement': improvement,
                'duration': duration,
                'optimal_cost': optimal_obj
            })

            # --- Complementarity Logic ---
            is_success = (improvement < 0)

            # Log attempts for previously failed neighborhoods in this cycle
            if failed_in_cycle:
                for failed in failed_in_cycle:
                    if failed not in t_store['attempts']: t_store['attempts'][failed] = {}
                    t_store['attempts'][failed][neigh] = t_store['attempts'][failed].get(neigh, 0) + 1

            if is_success:
                # Log success for previously failed neighborhoods
                for failed in failed_in_cycle:
                    if failed not in t_store['success']: t_store['success'][failed] = {}
                    t_store['success'][failed][neigh] = t_store['success'][failed].get(neigh, 0) + 1
                # Reset cycle on success
                failed_in_cycle = set()
            else:
                failed_in_cycle.add(neigh)

    except Exception as e:
        # print(f"Skipping {file_path}: {e}")
        pass

df = pd.DataFrame(data_rows)

if df.empty:
    print("No valid data found. Check file paths.")
else:
    print(f"   Loaded {len(df)} neighborhood calls.")

    print("Step 4/5: Calculating metrics...")
    # --- Metrics Calculation ---
    df['gap_reduction_pct'] = df.apply(lambda x: (abs(x['improvement']) / x['optimal_cost'] * 100), axis=1)
    df['roi'] = df.apply(lambda x: x['gap_reduction_pct'] / x['duration'] if x['duration'] > 1e-6 else 0, axis=1)
    df['is_success'] = df['improvement'] < 0

    # Sorting for consistent plotting
    sorted_neighs = sorted(list(unique_neighborhoods))
    sorted_topks = sorted(df['topk'].unique())
    sorted_types = sorted(df['instance_type'].unique())

    os.makedirs('results/artifacts', exist_ok=True)

    # ==========================================
    # 3. GENERATE GRAPHS
    # ==========================================
    print("Step 5/5: Generating graphs...")

    # --- Graph 1: Success Rate by Neighborhood (Grouped by Top-K) ---
    print("   Generating Graph 1 (Success Rate)...")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    succ_rate = df.groupby(['neighborhood', 'topk'])['is_success'].mean().reset_index()
    succ_rate['is_success'] *= 100  # Convert to %

    sns.barplot(data=succ_rate, x='neighborhood', y='is_success', hue='topk', palette='viridis', ax=ax1)
    set_plot_header(ax1, 'Improvement Success Rate by Neighborhood', 'Success Rate (%)', 'Neighborhood')
    plt.legend(title='Top-K', loc='upper right')
    plt.tight_layout()
    plt.savefig('results/artifacts/graph1_success_rate_topk.png')

    # --- Graph 2: Efficiency (ROI) by Neighborhood (Grouped by Top-K) ---
    print("   Generating Graph 2 (ROI per Neighborhood)...")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='neighborhood', y='roi', hue='topk', palette='viridis', ax=ax2, errorbar=None)
    set_plot_header(ax2, 'Efficiency (ROI) by Neighborhood', 'ROI (Gap % / Sec)', 'Neighborhood')
    plt.legend(title='Top-K')
    plt.tight_layout()
    plt.savefig('results/artifacts/graph2_roi_topk.png')

    # --- Graph 3: Success Rate by Instance Type (Grouped by Top-K) ---
    print("   Generating Graph 3 (Success Rate by Type)...")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    type_succ = df.groupby(['instance_type', 'topk'])['is_success'].mean().reset_index()
    type_succ['is_success'] *= 100

    sns.barplot(data=type_succ, x='instance_type', y='is_success', hue='topk', palette='viridis', ax=ax3)
    set_plot_header(ax3, 'Success Rate by Instance Type', 'Success Rate (%)', 'Instance Type')
    plt.legend(title='Top-K')
    plt.tight_layout()
    plt.savefig('results/artifacts/graph3_type_success_topk.png')

    # --- Graph 4: Complementarity Heatmap Grid ---
    print("   Generating Graph 4 (Heatmap Grid)...")
    n_rows = len(sorted_types)
    n_cols = len(sorted_topks)

    if n_rows > 0 and n_cols > 0:
        fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

        for r, i_type in enumerate(sorted_types):
            for c, k in enumerate(sorted_topks):
                ax = axes4[r, c]

                if i_type in transition_data and k in transition_data[i_type]:
                    t_data = transition_data[i_type][k]
                    # Create DF from counts
                    df_succ = pd.DataFrame(t_data['success']).reindex(index=sorted_neighs,
                                                                      columns=sorted_neighs).fillna(0)
                    df_atmpt = pd.DataFrame(t_data['attempts']).reindex(index=sorted_neighs,
                                                                        columns=sorted_neighs).fillna(0)

                    # Calculate Success Rate %
                    heatmap_val = (df_succ / df_atmpt).fillna(0) * 100

                    # Mask diagonal and cells with 0 attempts
                    mask = np.eye(len(sorted_neighs), dtype=bool) | (df_atmpt == 0)

                    sns.heatmap(heatmap_val, annot=True, fmt=".0f", cmap="YlGnBu",
                                vmin=0, vmax=100, square=True, cbar=False, ax=ax, mask=mask)

                    ax.set_title(f'{i_type} | Top-K={k}')
                    if r == n_rows - 1:
                        ax.set_xlabel('Succeeding')
                    else:
                        ax.set_xlabel('')
                    if c == 0:
                        ax.set_ylabel('Failed')
                    else:
                        ax.set_ylabel('')
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    ax.axis('off')

        plt.tight_layout()
        plt.savefig('results/artifacts/graph4_complementarity_grid.png')

    # --- Graph 5: Average Duration by Neighborhood (Grouped by Top-K) ---
    print("   Generating Graph 5 (Duration by Neighborhood)...")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='neighborhood', y='duration', hue='topk', palette='viridis', ax=ax5, errorbar=None)
    set_plot_header(ax5, 'Avg Duration by Neighborhood', 'Time (s)', 'Neighborhood')
    plt.legend(title='Top-K')
    plt.tight_layout()
    plt.savefig('results/artifacts/graph5_duration_topk.png')

    # --- Graph 6: ROI by Instance Type (Grouped by Top-K) ---
    print("   Generating Graph 6 (ROI by Type)...")
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='instance_type', y='roi', hue='topk', palette='viridis', ax=ax6, errorbar=None)
    set_plot_header(ax6, 'Efficiency (ROI) by Instance Type', 'ROI (Gap % / Sec)', 'Instance Type')
    plt.legend(title='Top-K')
    plt.tight_layout()
    plt.savefig('results/artifacts/graph6_roi_type_topk.png')

    # --- Graph 7: Total Time Distribution by Neighborhood ---
    print("   Generating Graph 7 (Total Time Distribution)...")
    # This might require summing duration per neighborhood per run first, but here we just show overall sum/distribution
    # Let's show boxplot of duration instead to show distribution
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='neighborhood', y='duration', hue='topk', palette='viridis', ax=ax7, showfliers=False)
    set_plot_header(ax7, 'Duration Distribution per Call', 'Time (s)', 'Neighborhood')
    plt.legend(title='Top-K')
    plt.tight_layout()
    plt.savefig('results/artifacts/graph7_duration_dist_topk.png')

    print("Analysis Complete. Check 'results/artifacts' folder.")