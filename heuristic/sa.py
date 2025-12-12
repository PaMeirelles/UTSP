import random
import math
from typing import List, Optional, Any

from heuristic.heuristic_tsp_solver import HeuristicTSPSolution
from heuristic.neighborhoods.shift import Shift
from heuristic.neighborhoods.switch import Switch
from heuristic.neighborhoods.two_opt import TwoOpt


class SimulatedAnnealing:

    def __init__(self,
                 solution: HeuristicTSPSolution,
                 initial_temp: float = 5000,
                 final_temp: float = 1,
                 cooling_rate: float = 0.99995,
                 neighborhood_classes: Optional[List[Any]]=None,
                 ):
        """
        Args:
            initial_temp: Starting temperature (T0).
            final_temp: Stopping temperature (Tf). Algorithm stops when T < Tf.
            cooling_rate: Factor to multiply T by at each step (0 < alpha < 1).
                          To ensure a smooth search, use a value close to 1 (e.g., 0.9995).
            neighborhood_classes: List of neighborhood strategies.
        """
        if neighborhood_classes is None:
            neighborhood_classes = [
            Shift,
            Switch,
            TwoOpt,
        ]
        self.solution = solution
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.neighborhood_classes = neighborhood_classes

    def solve(self, verbose: bool = False) -> HeuristicTSPSolution:
        if verbose:
            print(f"--- Starting Simulated Annealing ---")
            print(f"Params: T_start={self.initial_temp}, T_end={self.final_temp}, Alpha={self.cooling_rate}")

        # --- 1. Initialization ---
        s_current = self.solution.clone()
        s_best = s_current.clone()

        cost_current = s_current.get_solution_cost()
        cost_best = cost_current

        curr_temp = self.initial_temp
        iteration = 0

        # --- 2. Main Loop (Continuous Cooling) ---
        # We iterate until the temperature drops below the target
        while curr_temp > self.final_temp:
            iteration += 1

            # Select Neighborhood & Perturb
            neighborhood = random.choice(self.neighborhood_classes)(s_current)
            move = neighborhood.perturb()

            if not move:
                # No valid move found, cool down and continue
                curr_temp *= self.cooling_rate
                continue

            delta = neighborhood.evaluate(move)

            if delta == float('inf'):
                # Infeasible move, ignore but still cool down (time passes)
                curr_temp *= self.cooling_rate
                continue

            # Metropolis Acceptance Criterion
            accept = False
            if delta < 0:
                accept = True
            else:
                # Probability = exp(-delta / T)
                # Ensure T > 0 to avoid division by zero
                if curr_temp > 1e-9:
                    probability = math.exp(-delta / curr_temp)
                    if random.random() < probability:
                        accept = True
                else:
                    accept = False

            # Apply Move
            if accept:
                neighborhood.execute(move)
                cost_current += delta

                # Update Best?
                if cost_current < cost_best:
                    cost_best = cost_current
                    s_best = s_current.clone()
                    if verbose:
                        print(f"  -> New Best: {cost_best:.2f} (Iter: {iteration}, Temp: {curr_temp:.2f})")

            # --- COOLING STEP ---
            # 1 iteration per temp means we cool immediately
            curr_temp *= self.cooling_rate

            # Optional: Periodic log to show progress
            if verbose and iteration % 5000 == 0:
                print(f"     Iter {iteration}: Temp={curr_temp:.2f}, CurrCost={cost_current:.2f}")

        if verbose:
            print("--- SA Finished ---")
            print(f"Total Iterations: {iteration}")
            print(f"Best Cost: {cost_best}")

        return s_best
