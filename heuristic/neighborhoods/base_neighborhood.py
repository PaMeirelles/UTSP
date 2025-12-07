from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from heuristic.heuristic_tsp_solver import HeuristicTSPSolution

MoveArgs = TypeVar('MoveArgs')


class BaseNeighborhood(ABC, Generic[MoveArgs]):
    def __init__(self, solution: HeuristicTSPSolution) -> None:
        self.solution = solution
    @abstractmethod
    def evaluate(self, move_args: MoveArgs) -> float:
        """
        Calculates the cost of a potential move without
        permanently applying it to the solution.

        Returns:
            The cost (float) of the solution if this move
            were to be applied. Returns float('inf') if infeasible.
        """
        pass

    @abstractmethod
    def execute(self, move_args: MoveArgs) -> None:
        """
        Applies a given move to the neighborhood's solution,
        modifying it in-place.
        """
        pass

    @abstractmethod
    def perturb(self) -> MoveArgs | None:
        """
        Generates and applies one random, valid move.

        Returns:
            The MoveArgs of the move that was applied, or None
            if no move could be made.
        """
        pass

    @abstractmethod
    def search(self) -> MoveArgs | None:
        """
        Searches the neighborhood for an improving move. Uses the heatmap

        Returns:
            The MoveArgs of the best/first improving move, or None
            if no improving move is found.
        """
        pass