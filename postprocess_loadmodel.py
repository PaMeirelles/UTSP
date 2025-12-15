"""
Post-process results produced by `loadmodel.py` (the text output file).

This script parses the output file(s) and for each instance:
- reconstructs a dense heatmap from the saved top-k indices/values
- constructs a Hamiltonian tour greedily from edges sorted by heatmap value
- saves heatmap image and a plot of the tour over node coords
- writes the recovered tour and its length to a CSV

Usage:
    python postprocess_loadmodel.py --input results.txt --num_nodes 128 --topk 20 --distancetype EUC_2D

"""
import argparse
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import glob

try:
    from geopy.distance import geodesic
except Exception:
    geodesic = None


def parse_line(line, num_nodes=None, topk=None):
    parts = line.strip().split()
    # find markers
    try:
        output_idx = parts.index('output')
        indices_idx = parts.index('indices')
        value_idx = parts.index('value')
    except ValueError:
        raise ValueError('Input record missing required markers (output/indices/value)')

    coords_tokens = parts[:output_idx]
    if len(coords_tokens) % 2 != 0:
        raise ValueError(f'Coords token count is not even: {len(coords_tokens)}')
    inferred_num_nodes = len(coords_tokens) // 2

    # infer num_nodes and topk if not provided
    if num_nodes is None:
        num_nodes = inferred_num_nodes
    elif num_nodes != inferred_num_nodes:
        print(f'Warning: provided num_nodes={num_nodes} differs from inferred {inferred_num_nodes}; using inferred')
        num_nodes = inferred_num_nodes

    idx_tokens = parts[indices_idx + 1:value_idx]
    inferred_idx_count = len(idx_tokens)
    if inferred_idx_count == 0:
        raise ValueError('No index tokens found')
    if topk is None:
        if inferred_idx_count % num_nodes != 0:
            raise ValueError(f'Index token count {inferred_idx_count} not divisible by num_nodes {num_nodes}')
        topk = inferred_idx_count // num_nodes

    val_tokens = parts[value_idx + 1:]
    inferred_val_count = len(val_tokens)
    expected = num_nodes * topk
    if inferred_val_count < expected:
        raise ValueError(f'Not enough value tokens: expected {expected}, got {inferred_val_count}')

    coords = np.array([float(x) for x in coords_tokens[:2 * num_nodes]]).reshape((num_nodes, 2))

    out_tokens = parts[output_idx + 1:indices_idx]
    out = np.array([int(x) - 1 for x in out_tokens[:num_nodes]], dtype=int)

    idx_tokens = idx_tokens[:expected]
    indices = np.array([int(x) - 1 for x in idx_tokens], dtype=int).reshape((num_nodes, topk))

    val_tokens = val_tokens[:expected]
    values = np.array([float(x) for x in val_tokens]).reshape((num_nodes, topk))

    return coords, out, indices, values


def reconstruct_heatmap(indices, values, num_nodes, symmetric=False):
    H = np.zeros((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for k in range(indices.shape[1]):
            j = indices[i, k]
            if j < 0 or j >= num_nodes:
                continue
            v = values[i, k]
            if v > H[i, j]:
                H[i, j] = v
            if symmetric and v > H[j, i]:
                H[j, i] = v
    return H


def greedy_tour_from_heatmap(H):
    n = H.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = (H[i, j] + H[j, i]) / 2.0
            edges.append((w, i, j))
    edges.sort(reverse=True, key=lambda x: x[0])

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    degree = [0] * n
    adj = {i: [] for i in range(n)}
    added = 0
    for w, u, v in edges:
        if w <= 0:
            continue
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        fu = find(u)
        fv = find(v)
        if fu == fv:
            if added == n - 1:
                pass
            else:
                continue
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
        parent[fu] = fv
        added += 1
        if added == n:
            break

    start = 0
    for i in range(n):
        if len(adj[i]) == 1:
            start = i
            break
    tour = [start]
    prev = -1
    cur = start
    for _ in range(n - 1):
        neighs = adj[cur]
        nxt = None
        for nb in neighs:
            if nb != prev:
                nxt = nb
                break
        if nxt is None:
            return None
        tour.append(nxt)
        prev, cur = cur, nxt
    return tour


def tour_length(coords, tour, distancetype='EUC_2D'):
    n = len(tour)
    total = 0.0
    if distancetype == 'EUC_2D':
        for i in range(n):
            a = coords[tour[i]]
            b = coords[tour[(i + 1) % n]]
            total += np.linalg.norm(a - b)
    elif distancetype == 'GEO':
        if geodesic is None:
            raise RuntimeError('geopy not available')
        for i in range(n):
            a = coords[tour[i]]
            b = coords[tour[(i + 1) % n]]
            total += geodesic((float(a[1]), float(a[0])), (float(b[1]), float(b[0]))).meters
    elif distancetype == 'ATT':
        for i in range(n):
            a = coords[tour[i]]
            b = coords[tour[(i + 1) % n]]
            xd = a[0] - b[0]
            yd = a[1] - b[1]
            rij = math.sqrt((xd * xd + yd * yd) / 10.0)
            tij = round(rij)
            dij = tij + 1 if tij < rij else tij
            total += dij
    return total


def plot_heatmap(H, out_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(H, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_tour(coords, tour, out_path):
    tour_coords = np.array([coords[i] for i in tour] + [coords[tour[0]]])
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue')
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], c='red')
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=6)
    plt.title('Recovered Tour')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def detect_distancetype_from_filename(path):
    name = os.path.basename(path).lower()
    if 'geo' in name:
        return 'GEO'
    if 'att' in name:
        return 'ATT'
    return 'EUC_2D'


def record_iterator(fhandle, num_nodes=None, topk=None):
    """Yield complete logical records (possibly spanning multiple physical lines).

    Accumulate lines until we see the markers 'output', 'indices', 'value'
    and the expected counts of tokens for coords, indices and values. If
    `num_nodes` or `topk` are None they will be inferred from tokens when
    possible.
    """
    buf = ''
    for raw in fhandle:
        if not raw.strip():
            continue
        buf += ' ' + raw.strip()
        parts = buf.split()
        if not ('output' in parts and 'indices' in parts and 'value' in parts):
            continue
        try:
            out_idx = parts.index('output')
            ind_idx = parts.index('indices')
            val_idx = parts.index('value')
        except ValueError:
            continue

        coords_count = out_idx
        inferred_num_nodes = coords_count // 2 if coords_count % 2 == 0 and coords_count > 0 else None
        idx_count = ind_idx - (out_idx + 1)
        # try to infer num_nodes/topk if missing
        use_num = num_nodes if num_nodes is not None else inferred_num_nodes
        use_topk = topk
        if use_num is not None and idx_count > 0 and use_topk is None:
            if idx_count % use_num == 0:
                use_topk = idx_count // use_num

        val_count = len(parts) - (val_idx + 1)

        if use_num is None or use_topk is None:
            try:
                _ = parse_line(' '.join(parts), num_nodes, topk)
                record = ' '.join(parts)
                buf = ''
                yield record
            except Exception:
                continue
        else:
            expected_idx = use_num * use_topk
            if coords_count >= 2 * use_num and idx_count >= expected_idx and val_count >= expected_idx:
                needed_vals = (val_idx + 1) + expected_idx
                record_parts = parts[:needed_vals]
                record = ' '.join(record_parts)
                remaining = parts[needed_vals:]
                buf = ' '.join(remaining)
                yield record
            else:
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, nargs='+', help='Path(s) or glob(s) to loadmodel text output file(s)')
    parser.add_argument('--num_nodes', type=int, default=None, help='(Optional) override num_nodes; otherwise inferred per record')
    parser.add_argument('--topk', type=int, default=None, help='(Optional) override topk; otherwise inferred per record')
    parser.add_argument('--distancetype', type=str, default=None, choices=['EUC_2D', 'GEO', 'ATT', None])
    parser.add_argument('--outdir', '-o', default='postproc_outputs')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    input_paths = []
    for pattern in args.input:
        input_paths.extend(sorted(glob.glob(pattern)))
    if not input_paths:
        print('No input files matched the given patterns.')
        return

    overall_tours = []
    overall_lengths = []

    for filepath in input_paths:
        print(f'Processing {filepath}...')
        base = os.path.splitext(os.path.basename(filepath))[0]
        outdir = os.path.join(args.outdir, base)
        os.makedirs(outdir, exist_ok=True)
        tours = []
        lengths = []
        rec_idx = 0
        with open(filepath, 'r') as f:
            for record in record_iterator(f, args.num_nodes, args.topk):
                try:
                    coords, sol, indices, values = parse_line(record, args.num_nodes, args.topk)
                except Exception as e:
                    print(f'Skipping record {rec_idx}: parse error: {e}')
                    rec_idx += 1
                    continue

                H = reconstruct_heatmap(indices, values, indices.shape[0], symmetric=False)
                heat_path = os.path.join(outdir, f'heat_{rec_idx}.png')
                plot_heatmap(H, heat_path)

                tour = greedy_tour_from_heatmap(H)
                if tour is None:
                    # fallback: greedy nearest neighbor (use heatmap as similarity)
                    tour = [0]
                    visited = set(tour)
                    for _ in range(indices.shape[0] - 1):
                        cur = tour[-1]
                        candidates = [(H[cur, j], j) for j in range(indices.shape[0]) if j not in visited]
                        if not candidates:
                            break
                        nxt = max(candidates)[1]
                        tour.append(nxt)
                        visited.add(nxt)

                length = tour_length(coords, tour, distancetype=(args.distancetype or detect_distancetype_from_filename(filepath)))
                tours.append(tour)
                lengths.append(length)
                overall_tours.append(tour)
                overall_lengths.append(length)

                # save tour and plot
                tour_path = os.path.join(outdir, f'tour_{rec_idx}.png')
                plot_tour(coords, tour, tour_path)
                np.savetxt(os.path.join(outdir, f'tour_{rec_idx}.txt'), np.array(tour, dtype=int), fmt='%d')
                print(f'File {base} Instance {rec_idx}: tour length={length:.4f}, heatmap saved to {heat_path}, tour saved to {tour_path}')
                rec_idx += 1

        # write summary for this file
        import csv
        with open(os.path.join(outdir, 'summary.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['instance', 'tour_length'])
            for i, l in enumerate(lengths):
                writer.writerow([i, l])

    print('All files processed.')

    # write overall summary
    import csv
    with open(os.path.join(args.outdir, 'summary.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instance', 'tour_length'])
        for i, l in enumerate(overall_lengths):
            writer.writerow([i, l])


if __name__ == '__main__':
    main()
