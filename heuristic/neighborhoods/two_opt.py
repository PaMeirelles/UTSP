import random
from typing import Tuple, Literal
from dataclasses import dataclass
from heuristic.neighborhoods.base_neighborhood import BaseNeighborhood


@dataclass
class TwoOptArgs:
    i: int
    j: int


class TwoOpt(BaseNeighborhood[TwoOptArgs]):
    def evaluate(self, move_args: TwoOptArgs) -> float:
        """
        Evaluates reversing the segment between indices i+1 and j.
        move_args: (i, j) where i < j is generally expected but handled internally.
        """
        i, j = move_args.i, move_args.j
        tour = self.solution.tour
        n = len(tour)
        dist = self.solution.distances

        # 1. Canonicalize indices just in case
        if i > j:
            i, j = j, i

        # 2. Validity / No-op Checks
        # Adjacent nodes (j is immediately after i) -> No edge cut
        if j == i + 1:
            return self.solution.get_solution_cost()

        # Wrap-around adjacency (i=0, j=n-1) -> effectively no change in a closed tour
        if i == 0 and j == n - 1:
            return self.solution.get_solution_cost()

        # 3. Identify vertices
        # We are cutting edge (i, i+1) and (j, j+1)
        # We are adding edge (i, j) and (i+1, j+1)

        idx_i = i
        idx_i_next = (i + 1) % n
        idx_j = j
        idx_j_next = (j + 1) % n

        node_i = tour[idx_i]
        node_i_next = tour[idx_i_next]
        node_j = tour[idx_j]
        node_j_next = tour[idx_j_next]

        # 4. Calculate Delta
        # Edges removed
        removed = dist[node_i][node_i_next] + dist[node_j][node_j_next]

        # Edges added
        added = dist[node_i][node_j] + dist[node_i_next][node_j_next]

        return added - removed

    def execute(self, move_args: TwoOptArgs) -> None:
        """
        Executes the 2-opt move by reversing the segment tour[i+1 : j+1].
        """
        i, j = move_args.i, move_args.j
        if i > j:
            i, j = j, i

        tour = self.solution.tour

        # Python slice assignment handles the reversal efficiently
        # We reverse the segment starting AFTER i, up to and including j
        tour[i + 1: j + 1] = reversed(tour[i + 1: j + 1])
        self.solution.get_city_indexes()

    def perturb(self, max_trials: int = 20) -> TwoOptArgs | None:
        """
        Selects two random indices and performs a 2-opt reversal.
        """
        n = len(self.solution.tour)
        if n < 4:
            return None

        for _ in range(max_trials):
            i = random.randint(0, n - 3)
            j = random.randint(i + 2, n - 1)  # Ensure at least 1 node between them to not be adjacent

            # Avoid the wrap-around case (0 and n-1)
            if i == 0 and j == n - 1:
                continue

            move = TwoOptArgs(i, j)
            return move

        return None

    def search(self, improvement_mode: Literal["first", "best"] = "first") -> Tuple[TwoOptArgs, float] | None:
        """
        Searches for an improving 2-opt move using the nearest neighbor heatmap.

        Args:
            improvement_mode: "first" returns the first negative delta found.
                              "best" checks all heatmap candidates and returns the most negative delta.
        """
        tour = self.solution.tour
        n = len(tour)
        city_indices = self.solution.city_indices

        best_move = None
        best_delta = 1e-10 # previne q erros de arredondamento fodam a gnt

        for i in range(n):
            u = tour[i]

            # Check nearest neighbors of u to form a new connection (u, v)
            for v, _ in self.solution.edges[u]:
                if v == u: continue

                j = city_indices[v]

                # Canonicalize for the move format (low, high)
                idx1, idx2 = sorted((i, j))

                # Skip adjacent (no-op) and wrap-around cases
                if idx2 == idx1 + 1: continue
                if idx1 == 0 and idx2 == n - 1: continue

                move = TwoOptArgs(idx1, idx2)

                delta = self.evaluate(move)

                if delta < best_delta:
                    if improvement_mode == "first":
                        return move, delta

                    # Update best found
                    best_delta = delta
                    best_move = move

        if best_move is not None:
            return best_move, best_delta

        return None