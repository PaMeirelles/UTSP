import time
import copy
import random
import math
from dataclasses import dataclass
from typing import Any, List

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


def local_search(solution: HeuristicTSPSolution):
    """
    Executa busca local VND.
    """
    neighborhoods = [
            Shift,
            Switch,
            TwoOpt,
        ]
    k = 0
    calls: List[NeighborhoodCall] = []
    while k < len(neighborhoods):
        nb_class = neighborhoods[k]
        neighborhood = nb_class(solution)

        start = time.time()
        result = neighborhood.search()
        if result is not None:
            move, delta = neighborhood.search()
        else:
            move = None
            delta = 0
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
                 max_iter: int = 100,
                 perturbation_strength: int = 3,
                 initial_temp: float = 2000.0,
                 cooling_rate: float = 0.95):

        self.solution = solution
        self.max_iter = max_iter
        self.perturbation_strength = perturbation_strength
        self.initial_temp = initial_temp
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

    def run(self, report_stats: bool = False):
        start_time = time.time()
        print("--- Iniciando ILS Híbrido (JSON) ---")

        s_curr = self.solution.clone()

        # Chama a função simples
        calls = local_search(s_curr)
        self.calls += calls

        s_best = copy.deepcopy(s_curr)
        cost_best = s_best.get_solution_cost()
        cost_curr = cost_best

        temp = self.initial_temp
        print(f"Custo Inicial: {cost_best:.2f}")

        for i in range(self.max_iter):
            s_candidate = copy.deepcopy(s_curr)

            # Chama passando a força (o código lá em cima trata se for int)
            perturbation(s_candidate, self.perturbation_strength)
            calls = local_search(s_candidate)
            self.calls += calls

            cost_candidate = s_candidate.get_solution_cost()

            # Critério de Aceitação (SA Híbrido)
            delta = cost_candidate - cost_curr
            accept = False

            if delta < 0:
                accept = True
            else:
                if temp > 0.001:
                    try:
                        r = random.random()
                        if r < math.exp(-delta / temp):
                            accept = True
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
                    print(f"  -> Iter {i}: Novo melhor! {cost_best:.2f}")

            temp *= self.cooling_rate

        end_time = time.time()
        print("--- ILS Finalizado ---")
        print(f"Melhor custo: {cost_best:.2f}")
        print(f"Tempo: {end_time - start_time:.2f}s")
        if report_stats:
            self.report_stats()
        return s_best
