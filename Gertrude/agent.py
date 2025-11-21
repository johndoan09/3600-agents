from collections.abc import Callable
from typing import List, Set, Tuple

import numpy as np
from game import *

"""
A slightly smarter agent:

- Prefers laying eggs whenever possible (since eggs win the game).
- Otherwise, chooses between remaining valid moves using a simple scoring
  heuristic instead of purely random selection.

This version only uses methods we know exist from the starter code:
    - board.chicken_player.get_location()
    - board.get_valid_moves()

So it should be a safe drop-in improvement over the random baseline.
"""


class PlayerAgent:
    """
    You may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        # You can keep state here across turns if you want.
        # For now, we just track how many turns we've played.
        self.turn_index = 0

    # ------------- Internal helpers -------------

    def _move_type_name(self, move) -> str:
        """
        Robustly extract a string name for the move type.

        move is expected to be a (direction, move_type) tuple.
        We don't assume anything about the exact enum implementation:
        - If move_type has a .name attribute (Enum), use that.
        - Otherwise, fall back to str(move_type).
        """
        if not isinstance(move, (tuple, list)) or len(move) < 2:
            return ""

        move_type = move[1]
        if hasattr(move_type, "name"):
            return move_type.name
        return str(move_type)

    def _score_move(self, move) -> float:
        """
        Assign a numeric score to a move.

        High-level idea:
        - Egg moves are best (they directly increase our score).
        - Turd moves are useful, but less important than eggs.
        - Plain moves are the default fall-back.

        This can be expanded later (trapdoor avoidance, zoning, etc.).
        """
        move_type_name = self._move_type_name(move).lower()

        score = 0.0

        # Strongly prefer egg moves
        if "egg" in move_type_name:
            score += 100.0

        # Turds: still useful, but secondary
        elif "turd" in move_type_name:
            score += 10.0

        # Plain moves: fine, but not special
        else:
            score += 1.0

        # Tiny random jitter to break ties and avoid deterministic patterns
        score += np.random.random() * 0.01

        return score

    # ------------- Main decision function -------------

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        """
        Choose an action (direction, move_type) on your turn.

        Args:
            board: current Board state
            sensor_data: [(heard_white, felt_white), (heard_black, felt_black)]
            time_left: function that returns remaining thinking time (seconds)
        """
        self.turn_index += 1

        # Current location, mostly for debugging if you keep prints.
        location = board.chicken_player.get_location()

        # Get all legal moves from the engine.
        moves = board.get_valid_moves()

        # Safety check: if something weird happens and there are no valid moves
        # (the engine shouldn't let this happen), just return a random guess.
        if not moves:
            return moves[0] if moves else None

        # Score each move based on a simple heuristic.
        scored_moves = [(self._score_move(m), m) for m in moves]

        # Pick the move with the highest score.
        best_score = max(sm[0] for sm in scored_moves)
        best_candidates = [m for (s, m) in scored_moves if s == best_score]

        # Break ties randomly among equally good moves.
        chosen_move = best_candidates[np.random.randint(len(best_candidates))]

        return chosen_move
