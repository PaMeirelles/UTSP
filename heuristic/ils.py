import time
import copy
import random
import math
from dataclasses import dataclass
from typing import Any, List, Literal, Tuple, Dict

from heuristic.heuristic_tsp_solver import HeuristicTSPSolution
from heuristic.neighborhoods.base_neighborhood import BaseNeighborhood
from heuristic.neighborhoods.shift import Shift
from heuristic.neighborhoods.switch import Switch
from heuristic.neighborhoods.two_opt import TwoOpt


@dataclass
class NeighborhoodCall:
    duration: float
    neighborhood_type: BaseNeighborhood[Any]
    improvement: float


def local_search(solution: HeuristicTSPSolution, improvement_mode: Literal["first", "best"]):
    """
    Executa busca local VND.
    """
    neighborhoods = [
        Shift,
        Switch,
        TwoOpt,
    ]
    random.shuffle(neighborhoods)
    k = 0
    calls: List[NeighborhoodCall] = []
    while k < len(neighborhoods):
        nb_class = neighborhoods[k]
        neighborhood = nb_class(solution)

        start = time.time()
        result = neighborhood.search(improvement_mode)

        if result is not None:
            move, delta = result
        else:
            move = None
            delta = 0.0

        end = time.time()

        duration = end - start
        call = NeighborhoodCall(duration, neighborhood, delta)
        calls.append(call)

        if move:
            neighborhood.execute(move)
            k = 0
        else:
            k += 1
    return calls


def perturbation(solution: HeuristicTSPSolution, neighborhoods_arg):
    """
    Perturbação flexível para funcionar com ILS (3 args) e GA (4 args).
    """
    strength = neighborhoods_arg
    neighborhoods = [
        Switch,
        TwoOpt,
        Shift,
    ]

    for _ in range(strength):
        nb_class = random.choice(neighborhoods)
        neighborhood = nb_class(solution)

        move = neighborhood.perturb()
        if move:
            if neighborhood.evaluate(move) != float('inf'):
                neighborhood.execute(move)


# --- CLASSE ILS HÍBRIDA ---
class IteratedLocalSearch:
    def __init__(self,
                 solution: HeuristicTSPSolution,
                 max_iter: int = 50,
                 perturbation_strength: int = 3,
                 initial_temp: float = 1000.0,
                 final_temp: float = 0.001,  # Added argument to match Solver call
                 cooling_rate: float = 0.95,  # Kept for backward compat, but overridden below
                 improvement_mode: Literal["first", "best"] = "first"):

        self.solution = solution
        self.max_iter = max_iter
        self.perturbation_strength = perturbation_strength
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.improvement_mode = improvement_mode

        # --- CONCILIATION: Calculate Cooling Rate based on Max Iter ---
        # We want T to go from initial_temp to final_temp in exactly max_iter steps.
        # Formula: T_final = T_initial * (alpha ^ max_iter)
        # Therefore: alpha = (T_final / T_initial) ^ (1 / max_iter)

        if self.max_iter > 0 and self.initial_temp > 1e-9 and self.final_temp > 1e-9:
            try:
                self.cooling_rate = math.pow(self.final_temp / self.initial_temp, 1.0 / self.max_iter)
            except ValueError:
                self.cooling_rate = cooling_rate  # Fallback if math fails
        else:
            self.cooling_rate = cooling_rate

        self.calls: List[NeighborhoodCall] = []

    def report_stats(self):
        """
        By neighborhood, print number of calls, avg improvement, total improvement,
        avg time per call, and efficiency (ROI: Improvement per Second).
        """
        stats = {}

        for call in self.calls:
            name = type(call.neighborhood_type).__name__

            if name not in stats:
                stats[name] = {
                    'count': 0,
                    'total_improvement': 0.0,
                    'total_duration': 0.0
                }

            stats[name]['count'] += 1
            stats[name]['total_improvement'] += call.improvement
            stats[name]['total_duration'] += call.duration

        print("\n--- Statistics by Neighborhood ---")
        # Added Imp/s (ROI) column
        header = f"{'Neighborhood':<15} | {'Calls':<8} | {'Tot. Improv':<15} | {'Avg. Improv':<15} | {'Avg. Time (s)':<15} | {'Imp/s (ROI)':<15}"
        print(header)
        print("-" * len(header))

        for name, data in stats.items():
            count = data['count']
            total_imp = data['total_improvement']
            total_dur = data['total_duration']

            avg_imp = total_imp / count if count > 0 else 0.0
            avg_time = total_dur / count if count > 0 else 0.0

            # ROI calculation: Total Improvement / Total Time
            # If improvement is negative (cost reduction), this shows reduction per second.
            roi = total_imp / total_dur if total_dur > 0 else 0.0

            print(f"{name:<15} | {count:<8} | {total_imp:<15.4f} | {avg_imp:<15.4f} | {avg_time:<15.6f} | {roi:<15.4f}")
        print("----------------------------------\n")

    def run(self, verbose: bool = False) -> Tuple[List[int], List[NeighborhoodCall]]:
        start_time = time.time()
        my_id = random.randint(0, 1000)
        if verbose: print(f"--- Iniciando ILS Híbrido (Mode: {self.improvement_mode}) {my_id} ---")

        s_curr = self.solution.clone()

        # Chama a função simples
        calls = local_search(s_curr, improvement_mode=self.improvement_mode)
        self.calls += calls

        s_best = copy.deepcopy(s_curr)
        cost_best = s_best.get_solution_cost()
        cost_curr = cost_best

        temp = self.initial_temp
        if verbose:
            print(
                f"Custo Inicial: {cost_best:.3f} {my_id}, T0={temp:.4f}, Tf={self.final_temp:.6f}, alpha={self.cooling_rate:.6f}")

        for i in range(self.max_iter):
            if verbose:
                print(f"--- Iteration {i + 1} {my_id}---")

            s_candidate = copy.deepcopy(s_curr)

            # Chama passando a força (o código lá em cima trata se for int)
            perturbation(s_candidate, self.perturbation_strength)
            calls = local_search(s_candidate, improvement_mode=self.improvement_mode)
            self.calls += calls

            cost_candidate = s_candidate.get_solution_cost()

            # Critério de Aceitação (SA Híbrido)
            delta = cost_candidate - cost_curr
            accept = False

            if delta < 0:
                accept = True
            else:
                # Use dynamic final_temp instead of hardcoded 0.001
                if temp > self.final_temp:
                    try:
                        r = random.random()
                        if -delta / temp > -700:  # Overflow check
                            if r < math.exp(-delta / temp):
                                accept = True
                        else:
                            accept = False
                    except OverflowError:
                        accept = False
                else:
                    accept = False

            if accept:
                s_curr = s_candidate
                cost_curr = cost_candidate

                if cost_curr < cost_best:
                    s_best = copy.deepcopy(s_curr)
                    cost_best = cost_curr
                    if verbose:
                        print(f"  -> Iter {i}: Novo melhor! {cost_best:.3f} {my_id}")

            temp *= self.cooling_rate

        end_time = time.time()
        if verbose:
            print(f"--- ILS Finalizado  {my_id}---")
            print(f"Melhor custo: {cost_best:.2f} {my_id}")
            print(f"Tempo: {end_time - start_time:.2f}s {my_id}")
        if verbose:
            self.report_stats()
        return s_best, self.calls