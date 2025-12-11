import random
from typing import Tuple, Literal
from heuristic.neighborhoods.base_neighborhood import BaseNeighborhood
from dataclasses import dataclass

@dataclass
class ShiftArgs:
   source_idx: int
   target_idx: int


class Shift(BaseNeighborhood[ShiftArgs]):
    def evaluate(self, move_args: ShiftArgs) -> float:
        """
        Evaluates moving the city at source_idx to be inserted *before* target_idx.
        move_args: (source_idx, target_idx)
        """
        source_idx, target_idx = move_args.source_idx, move_args.target_idx
        tour = self.solution.tour
        n = len(tour)

        # 1. Validity checks
        # If target is same as source, or target is immediately after source (no-op)
        if source_idx == target_idx:
            return self.solution.get_solution_cost()

        # If target_idx is (source_idx + 1), we are inserting before the successor,
        # which is exactly where it already is.
        if target_idx == (source_idx + 1) % n:
            return self.solution.get_solution_cost()

        # 2. Identify nodes involved
        # Nodes around the source (being removed)
        u = tour[source_idx]
        u_prev = tour[(source_idx - 1) % n]
        u_next = tour[(source_idx + 1) % n]

        # Nodes around the target (insertion point)
        # We are inserting *before* tour[target_idx]
        v = tour[target_idx]
        v_prev = tour[(target_idx - 1) % n]

        # 3. Calculate Delta
        # Cost change = - (edges removed) + (edges added)

        # Removing u breaks (u_prev, u) and (u, u_next)
        # And adds (u_prev, u_next)
        # Note: We must handle the case where u_prev or u_next is part of the target edge
        # to avoid double counting, though strict index checks usually prevent this.

        dist = self.solution.distances

        current_cost = self.solution.get_solution_cost()

        # Subtraction (edges removed)
        delta = - dist[u_prev][u] - dist[u][u_next]

        # Addition (edge formed by closing the gap)
        delta += dist[u_prev][u_next]

        # Subtraction (edge broken at target)
        delta -= dist[v_prev][v]

        # Addition (edges formed by inserting u)
        delta += dist[v_prev][u] + dist[u][v]

        return delta

    def execute(self, move_args: ShiftArgs) -> None:
        """
        Executes the shift move.
        move_args: (source_idx, target_idx)
        """
        source_idx, target_idx = move_args.source_idx, move_args.target_idx
        tour = self.solution.tour

        # Retrieve the node
        node = tour[source_idx]

        # Remove the node
        # Note: If we pop source_idx, indices > source_idx shift down by 1.
        tour.pop(source_idx)

        # Calculate insertion index
        # If target_idx was greater than source_idx, it has shifted down by 1.
        insert_idx = target_idx
        if target_idx > source_idx:
            insert_idx -= 1

        # Insert the node
        tour.insert(insert_idx, node)
        self.solution.get_city_indexes()

    def perturb(self, max_trials=20) -> ShiftArgs | None:
        """
        Selects a random node and moves it to a random new position.
        """
        n = len(self.solution.tour)
        if n < 3:
            return None

        for _ in range(max_trials):
            source_idx = random.randint(0, n - 1)
            target_idx = random.randint(0, n - 1)

            move = ShiftArgs(source_idx, target_idx)

            # Check if valid no-op again to be safe, though execute handles it gracefully
            if source_idx == target_idx or target_idx == (source_idx + 1) % n:
                continue

            return move

        return None

    def search(self, improvement_mode: Literal["first", "best"] = "first") -> Tuple[ShiftArgs, float] | None:
        """
        Searches for an improving shift move using the heatmap (nearest neighbors).

        Args:
            improvement_mode: "first" returns the first negative delta found.
                              "best" checks all heatmap candidates and returns the most negative delta.
        """
        tour = self.solution.tour
        n = len(tour)
        city_indices = self.solution.city_indices

        best_move = None
        best_delta = 0.0

        # For every node u in the tour
        for i in range(n):
            u = tour[i]

            # Look at u's nearest neighbors (from heatmap/edges)
            for v, _ in self.solution.edges[u]:
                if v == u: continue

                j = city_indices[v]

                # --- Candidate 1: Insert u BEFORE v (target index j) ---
                if i != j and j != (i + 1) % n:
                    move = ShiftArgs(i, j)
                    delta = self.evaluate(move)

                    if delta < best_delta:
                        if improvement_mode == "first":
                            return move, delta
                        # Update best found so far
                        best_delta = delta
                        best_move = move

                # --- Candidate 2: Insert u AFTER v (target index j + 1) ---
                target_after = (j + 1) % n
                if i != target_after and target_after != (i + 1) % n:
                    move = ShiftArgs(i, target_after)
                    delta = self.evaluate(move)

                    if delta < best_delta:
                        if improvement_mode == "first":
                            return move, delta
                        # Update best found so far
                        best_delta = delta
                        best_move = move

        # If mode is "best" (or "first" finished without finding anything), return the best found
        if best_move is not None:
            return best_move, best_delta

        return None
