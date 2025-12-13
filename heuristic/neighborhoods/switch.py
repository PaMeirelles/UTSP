import random
from typing import Tuple, Literal
from dataclasses import dataclass
from heuristic.neighborhoods.base_neighborhood import BaseNeighborhood


@dataclass
class SwitchArgs:
    i: int
    j: int


class Switch(BaseNeighborhood[SwitchArgs]):
    def evaluate(self, move_args: SwitchArgs) -> float:
        """
        Evaluates swapping the cities at indices i and j.
        """
        i, j = move_args.i, move_args.j
        if i == j:
            return self.solution.get_solution_cost()

        tour = self.solution.tour
        n = len(tour)
        dist = self.solution.distances

        # For very small instances (n < 4), the adjacency logic can be tricky
        # (e.g., u_prev might be v), so we fall back to full calculation for safety.
        if n < 4:
            cost = 0.0
            for k in range(n):
                # Determine who is at position k after swap
                u = tour[j] if k == i else (tour[i] if k == j else tour[k])
                # Determine who is at position k+1 after swap
                next_k = (k + 1) % n
                v = tour[j] if next_k == i else (tour[i] if next_k == j else tour[next_k])

                cost += dist[u][v]
            return cost

        # Standard Delta Calculation
        u = tour[i]
        v = tour[j]

        # Identify neighbors in the current tour
        u_prev = tour[(i - 1) % n]
        u_next = tour[(i + 1) % n]
        v_prev = tour[(j - 1) % n]
        v_next = tour[(j + 1) % n]

        delta = 0.0

        # Case 1: Adjacent cities
        if (i + 1) % n == j:  # u is immediately before v
            # Edges removed: (u_prev, u), (u, v), (v, v_next)
            # Edges added:   (u_prev, v), (v, u), (u, v_next)
            delta -= (dist[u_prev][u] + dist[u][v] + dist[v][v_next])
            delta += (dist[u_prev][v] + dist[v][u] + dist[u][v_next])

        elif (j + 1) % n == i:  # v is immediately before u
            # Edges removed: (v_prev, v), (v, u), (u, u_next)
            # Edges added:   (v_prev, u), (u, v), (v, u_next)
            delta -= (dist[v_prev][v] + dist[v][u] + dist[u][u_next])
            delta += (dist[v_prev][u] + dist[u][v] + dist[v][u_next])

        # Case 2: Non-adjacent cities
        else:
            # Edges removed: (u_prev, u), (u, u_next), (v_prev, v), (v, v_next)
            # Edges added:   (u_prev, v), (v, u_next), (v_prev, u), (u, v_next)
            delta -= (dist[u_prev][u] + dist[u][u_next] + dist[v_prev][v] + dist[v][v_next])
            delta += (dist[u_prev][v] + dist[v][u_next] + dist[v_prev][u] + dist[u][v_next])

        return delta

    def execute(self, move_args: SwitchArgs) -> None:
        """
        Executes the swap move in place.
        """
        i, j = move_args.i, move_args.j
        self.solution.tour[i], self.solution.tour[j] = self.solution.tour[j], self.solution.tour[i]
        self.solution.get_city_indexes()

    def perturb(self, max_trials: int = 20) -> SwitchArgs | None:
        """
        Performs a random valid swap.
        """
        n = len(self.solution.tour)
        if n < 2:
            return None

        for _ in range(max_trials):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)

            if i == j:
                continue

            move = SwitchArgs(i, j)
            return move

        return None

    def search(self, improvement_mode: Literal["first", "best"] = "first") -> Tuple[SwitchArgs, float] | None:
        """
        Searches for an improving swap move using heatmap.

        Args:
            improvement_mode: "first" returns the first negative delta found.
                              "best" checks all heatmap candidates and returns the most negative delta.
        """
        tour = self.solution.tour
        n = len(tour)
        city_indices = self.solution.city_indices

        best_move = None
        best_delta = -1e-10 # previne q erros de arredondamento fodam a gnt

        for i in range(n):
            u = tour[i]
            # Optimization: Only try swapping u with its nearest neighbors (v)
            for v, _ in self.solution.edges[u]:
                if v == u: continue

                # Look up where neighbor v is currently located in the tour
                j = city_indices[v]

                if i == j: continue

                move = SwitchArgs(i, j)
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