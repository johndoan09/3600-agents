from collections.abc import Callable
from typing import List, Tuple, Optional

import numpy as np
from game import *


"""
Minim (Option C): a utility-maximizing chicken agent with

- Trapdoor belief tracking (white + black).
- Sensor-based Bayesian updates using (heard, felt) probabilities.
- Trapdoor risk avoidance in move utility.
- Position evaluation (egg-parity + distance to center + risk).
- Alpha–beta minimax with:
    * iterative deepening at the root
    * move ordering based on immediate utility
    * robust handling of forecast_move()/reverse_perspective()

The agent tries to maximize:
    immediate_move_utility + gamma * (future position value)
assuming the opponent plays adversarially.
"""


class PlayerAgent:
    """
    __init__ and play are the entry points used by the engine.
    You may add other helper methods as needed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        # Board parameters (fixed by assignment: 8x8)
        self.board_size = 8
        self.center_x = (self.board_size - 1) / 2.0
        self.center_y = (self.board_size - 1) / 2.0

        # Our egg parity: 0 for even squares, 1 for odd squares, None if unknown.
        # We infer this the first time we see a legal Egg move.
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs (for each color separately).
        # belief_white[y, x] = P(white trapdoor at (x, y))
        # belief_black[y, x] = P(black trapdoor at (x, y))
        self.belief_white = np.zeros(
            (self.board_size, self.board_size), dtype=float)
        self.belief_black = np.zeros(
            (self.board_size, self.board_size), dtype=float)
        self._init_trapdoor_beliefs()

        # How bad is it to step on a trapdoor? (They get 4 eggs + you get reset.)
        self.trapdoor_penalty = 500.0

        # Discount factor for future position evaluation
        self.gamma = 0.5

        # Turn counter (not strictly needed, but handy if you want to time-budget)
        self.turn_index = 0

    # -------------------------------------------------------------------------
    # Initialization of trapdoor beliefs
    # -------------------------------------------------------------------------

    def _init_trapdoor_beliefs(self):
        """
        Initialize beliefs approximately uniformly over all squares of the
        correct color for each trapdoor (one white, one black).
        """
        # White trapdoor: even parity squares (i + j is even)
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 0:
                    self.belief_white[y, x] = 1.0
        total_white = self.belief_white.sum()
        if total_white > 0:
            self.belief_white /= total_white

        # Black trapdoor: odd parity squares (i + j is odd)
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 1:
                    self.belief_black[y, x] = 1.0
        total_black = self.belief_black.sum()
        if total_black > 0:
            self.belief_black /= total_black

    # -------------------------------------------------------------------------
    # Helpers for enums and moves
    # -------------------------------------------------------------------------

    def _unpack_move(self, move):
        """
        Safely unpack (direction, move_type) from the move tuple.
        """
        if not isinstance(move, (tuple, list)) or len(move) < 2:
            return None, None
        return move[0], move[1]

    def _enum_name(self, obj) -> str:
        """
        Get the .name of an Enum-like object if available, else its string.
        """
        if hasattr(obj, "name"):
            return obj.name
        return str(obj)

    # -------------------------------------------------------------------------
    # Trapdoor belief update using sensor_data
    # -------------------------------------------------------------------------

    def _sensor_likelihoods_for_distance(self, dx: int, dy: int) -> Tuple[float, float]:
        """
        Given dx, dy from current square to a candidate trapdoor square,
        return (p_hear, p_feel) according to the rules:

        - Share an edge: 50% hear, 30% feel
        - Diagonal: 25% hear, 15% feel
        - Square that shares an edge with either of the above ("second ring"):
          10% hear, 0% feel
        - Elsewhere: 0% hear, 0% feel
        """
        ax, ay = abs(dx), abs(dy)
        # Edge neighbor (up/down/left/right)
        if (ax == 1 and ay == 0) or (ax == 0 and ay == 1):
            return 0.5, 0.3
        # Diagonal neighbor
        if ax == 1 and ay == 1:
            return 0.25, 0.15
        # Second ring (approximate): Chebyshev distance 2
        if max(ax, ay) == 2:
            return 0.10, 0.0
        # Otherwise, essentially no signal
        return 0.0, 0.0

    def _update_trapdoor_belief_for_color(
        self,
        belief: np.ndarray,
        parity: int,
        location: Tuple[int, int],
        heard: bool,
        felt: bool,
    ):
        """
        Update the belief grid for a single trapdoor color using Bayes:
        P(T | sensor) ∝ P(sensor | T) * P(T).

        Only squares with the given parity are valid candidates for this color.
        """
        x0, y0 = location
        height, width = belief.shape

        new_belief = np.zeros_like(belief)

        for y in range(height):
            for x in range(width):
                if (x + y) % 2 != parity:
                    # This color's trapdoor cannot be here; probability stays 0.
                    continue

                prior = belief[y, x]
                if prior <= 0.0:
                    continue

                dx = x - x0
                dy = y - y0
                p_hear, p_feel = self._sensor_likelihoods_for_distance(dx, dy)

                # Likelihood for hearing
                lh_hear = p_hear if heard else (1.0 - p_hear)
                # Likelihood for feeling
                lh_feel = p_feel if felt else (1.0 - p_feel)

                likelihood = lh_hear * lh_feel
                new_belief[y, x] = prior * likelihood

        total = new_belief.sum()
        if total > 1e-12:
            new_belief /= total
            belief[:, :] = new_belief

        # If we are currently standing on a square of this color and haven't fallen
        # through, we can safely set its probability to zero.
        if (x0 + y0) % 2 == parity:
            belief[y0, x0] = 0.0
            s = belief.sum()
            if s > 0:
                belief[:, :] = belief / s

    def _update_trapdoor_beliefs(
        self,
        location: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]],
    ):
        """
        Update beliefs for white and black trapdoors using the latest sensor data.
        sensor_data:
            sensor_data[0] = (heard_white, felt_white)
            sensor_data[1] = (heard_black, felt_black)
        """
        (heard_white, felt_white) = sensor_data[0]
        (heard_black, felt_black) = sensor_data[1]

        # White squares: parity 0
        self._update_trapdoor_belief_for_color(
            self.belief_white,
            parity=0,
            location=location,
            heard=heard_white,
            felt=felt_white,
        )

        # Black squares: parity 1
        self._update_trapdoor_belief_for_color(
            self.belief_black,
            parity=1,
            location=location,
            heard=heard_black,
            felt=felt_black,
        )

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    def _apply_direction(self, location: Tuple[int, int], direction) -> Tuple[int, int]:
        """
        Approximate the resulting (x, y) location after moving in `direction`
        from `location`.

        Assumes:
        - (0,0) is top-left
        - "Up" decreases y, "Down" increases y
        - "Left" decreases x, "Right" increases x
        """
        x, y = location
        dir_name = self._enum_name(direction).lower()

        dx, dy = 0, 0
        if "up" in dir_name:
            dy = -1
        elif "down" in dir_name:
            dy = 1
        elif "left" in dir_name:
            dx = -1
        elif "right" in dir_name:
            dx = 1

        return x + dx, y + dy

    def _trapdoor_risk_at(self, x: int, y: int) -> float:
        """
        Probability that (x, y) is any trapdoor (white or black).
        """
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return 0.0
        return float(self.belief_white[y, x] + self.belief_black[y, x])

    def _maybe_update_egg_parity(
        self,
        location: Tuple[int, int],
        moves: List[Tuple],
    ):
        """
        Infer whether we are Chicken A (even parity) or Chicken B (odd parity)
        the first time we see a legal Egg move.
        """
        if self.my_egg_parity is not None:
            return

        x, y = location
        for move in moves:
            _, move_type = self._unpack_move(move)
            move_type_name = self._enum_name(move_type).lower()
            if "egg" in move_type_name:
                self.my_egg_parity = (x + y) % 2
                break

    def _immediate_utility_of_move(
        self,
        move,
        board_state: board.Board,
        sensor_data: List[Tuple[bool, bool]],
    ) -> float:
        """
        Immediate utility from executing this move from the current board_state:
        - +100 for Egg
        - +15 for Turd
        - +1 for Plain
        - +3 if destination square matches our egg-parity (once known)
        - - trapdoor_penalty * P(stepping on trapdoor at destination)
        """
        direction, move_type = self._unpack_move(move)
        if direction is None or move_type is None:
            return -1e9

        move_type_name = self._enum_name(move_type).lower()
        utility = 0.0

        # Base utility from move type.
        if "egg" in move_type_name:
            utility += 100.0
        elif "turd" in move_type_name:
            utility += 15.0
        else:
            utility += 1.0

        # Destination-based features
        current_loc = board_state.chicken_player.get_location()
        dest_x, dest_y = self._apply_direction(current_loc, direction)

        # Mild bonus for landing on "our" egg-color squares (if parity known).
        if self.my_egg_parity is not None:
            if (dest_x + dest_y) % 2 == self.my_egg_parity:
                utility += 3.0

        # Trapdoor risk penalty at the destination square.
        risk = self._trapdoor_risk_at(dest_x, dest_y)
        utility -= self.trapdoor_penalty * risk

        # Tiny jitter to break ties.
        utility += np.random.random() * 0.01

        return utility

    # -------------------------------------------------------------------------
    # Board evaluation for alpha–beta leaves
    # -------------------------------------------------------------------------

    def _evaluate_board(self, board_state: board.Board) -> float:
        """
        Lightweight static evaluation for board_state.

        Uses:
        - Bonus if our position matches our egg-parity (good for future eggs).
        - Mild penalty for standing very close to the board center
          (trapdoors more likely near center).
        - Mild penalty for being on a square with high trapdoor risk.
        """
        loc = board_state.chicken_player.get_location()
        x, y = loc

        value = 0.0

        # Egg-parity bonus (if known).
        if self.my_egg_parity is not None:
            if (x + y) % 2 == self.my_egg_parity:
                value += 5.0

        # Mild penalty for being near the center.
        dx = x - self.center_x
        dy = y - self.center_y
        dist2 = dx * dx + dy * dy
        center_penalty = max(0.0, 6.0 - dist2)  # only near the center
        value -= 0.5 * center_penalty

        # Penalty for standing directly on a risky square (if belief says so).
        risk_here = self._trapdoor_risk_at(x, y)
        value -= 0.5 * self.trapdoor_penalty * risk_here

        return value

    def _minimax_leaf_value(self, board_state: Optional[board.Board]) -> float:
        """
        Static evaluation of a board position for use at the leaves of alpha-beta search.
        If board_state is None (e.g., terminal from forecast_move), return neutral 0.
        """
        if board_state is None:
            return 0.0
        return self._evaluate_board(board_state)

    # -------------------------------------------------------------------------
    # Alpha–beta minimax search (with move ordering)
    # -------------------------------------------------------------------------

    def _alpha_beta(
        self,
        board_state: Optional[board.Board],
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        time_left: Callable,
        sensor_data: List[Tuple[bool, bool]],
    ) -> float:
        """
        Alpha-beta minimax search.

        Assumptions:
        - board_state is from the perspective of the player who is about to move.
        - maximizing_player == True  -> it's *our* turn (we maximize eval).
        - maximizing_player == False -> it's opponent's turn (we assume
          they minimize eval).
        """
        # If board_state is None, treat as leaf with neutral evaluation.
        if board_state is None:
            return self._minimax_leaf_value(None)

        # Time safety: if we're low on time, treat as leaf.
        try:
            if time_left() < 0.5:
                return self._minimax_leaf_value(board_state)
        except Exception:
            # If time_left misbehaves, ignore and continue.
            pass

        moves = board_state.get_valid_moves()
        if depth == 0 or not moves:
            return self._minimax_leaf_value(board_state)

        # Move ordering: sort moves by immediate utility (desc for max, asc for min).
        scored_moves = [
            (self._immediate_utility_of_move(m, board_state, sensor_data), m)
            for m in moves
        ]
        if maximizing_player:
            scored_moves.sort(key=lambda x: x[0], reverse=True)
        else:
            scored_moves.sort(key=lambda x: x[0])

        if maximizing_player:
            value = -1e9
            expanded = False
            for _, move in scored_moves:
                direction, move_type = self._unpack_move(move)
                if direction is None or move_type is None:
                    continue

                # Apply our move
                try:
                    next_board = board_state.forecast_move(
                        direction, move_type)
                except (AttributeError, TypeError):
                    continue

                if next_board is None:
                    continue

                # Now it's the opponent's turn -> reverse perspective if available
                try:
                    next_board = next_board.reverse_perspective()
                except AttributeError:
                    pass

                expanded = True
                child_value = self._alpha_beta(
                    next_board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing_player=False,
                    time_left=time_left,
                    sensor_data=sensor_data,
                )
                if child_value > value:
                    value = child_value
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break  # beta cut-off

            if not expanded:
                return self._minimax_leaf_value(board_state)
            return value
        else:
            # Opponent node (they minimize our eval)
            value = 1e9
            expanded = False
            for _, move in scored_moves:
                direction, move_type = self._unpack_move(move)
                if direction is None or move_type is None:
                    continue

                # Apply opponent's move
                try:
                    next_board = board_state.forecast_move(
                        direction, move_type)
                except (AttributeError, TypeError):
                    continue

                if next_board is None:
                    continue

                # Now it's our turn -> reverse perspective if available
                try:
                    next_board = next_board.reverse_perspective()
                except AttributeError:
                    pass

                expanded = True
                child_value = self._alpha_beta(
                    next_board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing_player=True,
                    time_left=time_left,
                    sensor_data=sensor_data,
                )
                if child_value < value:
                    value = child_value
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break  # alpha cut-off

            if not expanded:
                return self._minimax_leaf_value(board_state)
            return value

    # -------------------------------------------------------------------------
    # Root move selection with iterative deepening alpha–beta
    # -------------------------------------------------------------------------

    def _choose_move_alpha_beta(
        self,
        board_state: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        """
        Root-level move selection using iterative deepening alpha-beta search.

        - Adapts depth based on remaining time.
        - At the root, we are always the maximizing player.
        - For each candidate move:
            total_value = immediate_utility(move) + gamma * alpha_beta(child)
        """
        moves = board_state.get_valid_moves()
        if not moves:
            return moves[0] if moves else None

        # Base move ordering for root (helps all depths).
        base_scored_moves = [
            (self._immediate_utility_of_move(m, board_state, sensor_data), m)
            for m in moves
        ]
        base_scored_moves.sort(key=lambda x: x[0], reverse=True)

        # Decide maximum depth based on remaining time.
        try:
            remaining_time = time_left()
        except Exception:
            remaining_time = 9999

        if remaining_time > 180:
            max_depth = 4
        elif remaining_time > 60:
            max_depth = 3
        else:
            max_depth = 2

        best_move_overall = base_scored_moves[0][1]
        best_value_overall = -1e9

        # Iterative deepening: depth 1 up to max_depth
        for depth in range(1, max_depth + 1):
            try:
                if time_left() < 1.0:
                    break
            except Exception:
                pass

            best_value_this_depth = -1e9
            best_moves_this_depth: List[Tuple] = []

            for _, move in base_scored_moves:
                direction, move_type = self._unpack_move(move)
                if direction is None or move_type is None:
                    continue

                # Immediate utility from our perspective.
                immediate = self._immediate_utility_of_move(
                    move, board_state, sensor_data
                )

                # Apply our move
                try:
                    next_board = board_state.forecast_move(
                        direction, move_type)
                except (AttributeError, TypeError):
                    # If forecast_move fails, treat this move as immediate-only.
                    total_val = immediate
                    if total_val > best_value_this_depth:
                        best_value_this_depth = total_val
                        best_moves_this_depth = [move]
                    elif total_val == best_value_this_depth:
                        best_moves_this_depth.append(move)
                    continue

                if next_board is None:
                    # Terminal/invalid state we can't expand; just use immediate utility.
                    total_val = immediate
                    if total_val > best_value_this_depth:
                        best_value_this_depth = total_val
                        best_moves_this_depth = [move]
                    elif total_val == best_value_this_depth:
                        best_moves_this_depth.append(move)
                    continue

                # Opponent's turn next -> reverse perspective if available
                try:
                    next_board = next_board.reverse_perspective()
                except AttributeError:
                    pass

                # Future value: opponent to move, so minimizing player.
                future_value = self._alpha_beta(
                    next_board,
                    depth=depth - 1,
                    alpha=-1e9,
                    beta=1e9,
                    maximizing_player=False,
                    time_left=time_left,
                    sensor_data=sensor_data,
                )

                total_value = immediate + self.gamma * future_value

                if total_value > best_value_this_depth:
                    best_value_this_depth = total_value
                    best_moves_this_depth = [move]
                elif total_value == best_value_this_depth:
                    best_moves_this_depth.append(move)

            # If we found any moves at this depth, update overall best.
            if best_moves_this_depth:
                best_move_overall = best_moves_this_depth[
                    np.random.randint(len(best_moves_this_depth))
                ]
                best_value_overall = best_value_this_depth

        # Final choice from best_move_overall
        return best_move_overall

    # -------------------------------------------------------------------------
    # Main decision function
    # -------------------------------------------------------------------------

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        """
        Choose an action (direction, move_type) to maximize total utility
        using trapdoor beliefs + iterative deepening alpha-beta search.
        """
        self.turn_index += 1

        # Get all legal moves.
        moves = board.get_valid_moves()
        if not moves:
            return moves[0] if moves else None

        # Learn our egg-parity if we can.
        current_loc = board.chicken_player.get_location()
        self._maybe_update_egg_parity(current_loc, moves)

        # Update trapdoor beliefs using the latest sensor data.
        self._update_trapdoor_beliefs(current_loc, sensor_data)

        # Use alpha-beta search to pick a move.
        chosen_move = self._choose_move_alpha_beta(
            board, sensor_data, time_left)
        return chosen_move
