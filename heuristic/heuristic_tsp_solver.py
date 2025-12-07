from typing import List, Optional, Tuple


class HeuristicTSPSolution:
    def __init__(self, distances: List[List[float]], heatmap: List[List[float]], top_k: int) -> None:
        self.distances: List[List[float]] = distances

        # Validation checks
        if top_k > len(self.distances):
            raise ValueError("Top k must be less than the number of distances")

        if len(self.distances) != len(heatmap):
            raise ValueError("The number of distances must match the number of heatmap")

        for i in range(len(self.distances)):
            if len(self.distances[i]) != len(heatmap[i]):
                raise ValueError("The number of distances must match the number of heatmap")

        self.heatmap: List[List[float]] = heatmap
        self.top_k: int = top_k
        self.tour: Optional[List[int]] = None
        self.edges: List[List[Tuple[int, float]]] = []

        for i in range(len(self.distances)):
            # Create tuples of (index, distance)
            # We filter out the node itself (distance 0 to self) to avoid self-loops
            row_dists = [(idx, d) for idx, d in enumerate(self.distances[i]) if idx != i]

            row_dists.sort(key=lambda x: x[1])

            self.edges.append(row_dists[:top_k])

        self.city_indices = None

    def clone(self):
        clone = HeuristicTSPSolution(self.distances, self.heatmap, self.top_k)
        clone.tour = self.tour
        clone.edges = self.edges
        clone.city_indices = self.city_indices
        return clone


    def get_city_indexes(self) -> None:
        if self.tour is not None:
            self.city_indices = {city: idx for idx, city in enumerate(self.tour)}

    def construct_solution(self, start_node: int = 0) -> None:
        """
        Builds a solution using the Cheapest Insertion heuristic.
        It starts with a minimal tour and iteratively inserts the unvisited node 
        that causes the smallest increase in total tour length.
        """
        num_cities = len(self.distances)
        unvisited = set(range(num_cities))

        # Step 1: Initialize tour with the start node
        tour = [start_node]
        unvisited.remove(start_node)

        # Step 2: Find the nearest neighbor to the start node to form the initial edge
        # We use self.edges (top_k) for speed, but fallback to all distances if needed
        nearest_neighbor = self.edges[start_node][0][0]
        tour.append(nearest_neighbor)
        unvisited.remove(nearest_neighbor)

        # Step 3: Iterate until all cities are visited
        while unvisited:
            best_city = -1
            best_position = -1
            min_increase = float('inf')

            # For every unvisited city, find the best insertion point in the current tour
            for city in unvisited:
                for i in range(len(tour)):
                    u = tour[i]
                    v = tour[(i + 1) % len(tour)]  # Edge (u, v)

                    # Cost to insert 'city' between u and v:
                    # d(u, city) + d(city, v) - d(u, v)
                    dist_u_city = self.distances[u][city]
                    dist_city_v = self.distances[city][v]
                    dist_u_v = self.distances[u][v]

                    increase = dist_u_city + dist_city_v - dist_u_v

                    if increase < min_increase:
                        min_increase = increase
                        best_city = city
                        best_position = i + 1

            # Insert the best city found into the best position
            tour.insert(best_position, best_city)
            unvisited.remove(best_city)

        self.tour = tour
        self.get_city_indexes()

    def get_solution_cost(self) -> float:
        """Calculates the total distance of the generated solution."""
        if not self.tour:
            return 0.0

        cost = 0.0
        for i in range(len(self.tour)):
            u = self.tour[i]
            v = self.tour[(i + 1) % len(self.tour)]
            cost += self.distances[u][v]
        return cost
