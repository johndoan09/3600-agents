from collections.abc import Callable
from typing import List, Set, Tuple, Optional

import numpy as np
from game import *


"""
Exploration-heavy, egg-greedy agent.

Key ideas:
- Track trapdoor beliefs, but only avoid very high-risk squares.
- Greedy for EGG moves: if an egg is available, always choose an egg move.
- Strong anti-camping:
    * reward moving far from starting corner
    * penalize revisiting recent positions
    * penalize standing on squares where we've already laid eggs
- Use alpha–beta (with iterative deepening and move ordering) for non-egg moves.
"""


class PlayerAgent:
    """
    __init__ and play are the entry points used by the engine.
    You may add other helper methods as needed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        # Board parameters (fixed: 8x8)
        self.board_size = 8
        self.center_x = (self.board_size - 1) / 2.0
        self.center_y = (self.board_size - 1) / 2.0

        # Our starting location (for exploration reward)
        self.start_x, self.start_y = board.chicken_player.get_location()

        # Our egg parity: 0 for even squares, 1 for odd squares, None if unknown.
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs (one white, one black).
        self.belief_white = np.zeros(
            (self.board_size, self.board_size), dtype=float)
        self.belief_black = np.zeros(
            (self.board_size, self.board_size), dtype=float)
        self._init_trapdoor_beliefs()

        # Stepping on trapdoor penalty – lowered so we’re not terrified.
        self.trapdoor_penalty = 150.0

        # Future discount for minimax
        self.gamma = 0.5

        # Turn counter
        self.turn_index = 0

        # Anti-camping memory
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 6
        self.egg_squares: Set[Tuple[int, int]] = set()

    # -------------------------------------------------------------------------
    # Trapdoor beliefs
    # -------------------------------------------------------------------------

    def _init_trapdoor_beliefs(self):
        # White trapdoor: even parity squares
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 0:
                    self.belief_white[y, x] = 1.0
        s = self.belief_white.sum()
        if s > 0:
            self.belief_white /= s

        # Black trapdoor: odd parity squares
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 1:
                    self.belief_black[y, x] = 1.0
        s = self.belief_black.sum()
        if s > 0:
            self.belief_black /= s

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _unpack_move(self, move):
        if not isinstance(move, (tuple, list)) or len(move) < 2:
            return None, None
        return move[0], move[1]

    def _enum_name(self, obj) -> str:
        return obj.name if hasattr(obj, "name") else str(obj)

    # -------------------------------------------------------------------------
    # Sensor → belief update
    # -------------------------------------------------------------------------

    def _sensor_likelihoods_for_distance(self, dx: int, dy: int) -> Tuple[float, float]:
        ax, ay = abs(dx), abs(dy)
        # Edge neighbor
        if (ax == 1 and ay == 0) or (ax == 0 and ay == 1):
            return 0.5, 0.3
        # Diagonal
        if ax == 1 and ay == 1:
            return 0.25, 0.15
        # Second ring (Chebyshev distance 2)
        if max(ax, ay) == 2:
            return 0.10, 0.0
        return 0.0, 0.0

    def _update_trapdoor_belief_for_color(
        self,
        belief: np.ndarray,
        parity: int,
        location: Tuple[int, int],
        heard: bool,
        felt: bool,
    ):
        x0, y0 = location
        h, w = belief.shape
        new_belief = np.zeros_like(belief)

        for y in range(h):
            for x in range(w):
                if (x + y) % 2 != parity:
                    continue
                prior = belief[y, x]
                if prior <= 0.0:
                    continue

                dx, dy = x - x0, y - y0
                p_hear, p_feel = self._sensor_likelihoods_for_distance(dx, dy)

                lh_hear = p_hear if heard else (1.0 - p_hear)
                lh_feel = p_feel if felt else (1.0 - p_feel)
                likelihood = lh_hear * lh_feel

                new_belief[y, x] = prior * likelihood

        total = new_belief.sum()
        if total > 1e-12:
            new_belief /= total
            belief[:, :] = new_belief

        # If we are standing on a square of this color and didn't fall,
        # that square can't be the trapdoor.
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
        (heard_white, felt_white) = sensor_data[0]
        (heard_black, felt_black) = sensor_data[1]

        self._update_trapdoor_belief_for_color(
            self.belief_white, parity=0, location=location,
            heard=heard_white, felt=felt_white
        )
        self._update_trapdoor_belief_for_color(
            self.belief_black, parity=1, location=location,
            heard=heard_black, felt=felt_black
        )

    # -------------------------------------------------------------------------
    # Movement + risk helpers
    # -------------------------------------------------------------------------

    def _apply_direction(self, location: Tuple[int, int], direction) -> Tuple[int, int]:
        x, y = location
        name = self._enum_name(direction).lower()
        dx = dy = 0
        if "up" in name:
            dy = -1
        elif "down" in name:
            dy = 1
        elif "left" in name:
            dx = -1
        elif "right" in name:
            dx = 1
        return x + dx, y + dy

    def _trapdoor_risk_at(self, x: int, y: int) -> float:
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return 0.0
        return float(self.belief_white[y, x] + self.belief_black[y, x])

    def _maybe_update_egg_parity(
        self,
        location: Tuple[int, int],
        moves: List[Tuple],
    ):
        if self.my_egg_parity is not None:
            return
        x, y = location
        for move in moves:
            _, mt = self._unpack_move(move)
            if mt is None:
                continue
            if "egg" in self._enum_name(mt).lower():
                self.my_egg_parity = (x + y) % 2
                break

    # -------------------------------------------------------------------------
    # Anti-camping / exploration features
    # -------------------------------------------------------------------------

    def _corner_camping_penalty(self, x: int, y: int) -> float:
        d_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
        if d_edge <= 0:
            return 3.0
        if d_edge == 1:
            return 1.5
        return 0.0

    def _recent_position_penalty(self, x: int, y: int) -> float:
        penalty = 0.0
        for (px, py) in self.recent_positions:
            d = abs(px - x) + abs(py - y)
            if d == 0:
                penalty += 4.0
            elif d == 1:
                penalty += 2.0
        return penalty

    def _distance_from_start_bonus(self, x: int, y: int) -> float:
        """
        Reward being far from our starting corner to encourage exploration.
        """
        d = abs(x - self.start_x) + abs(y - self.start_y)
        return 0.6 * d   # scaled so it's meaningful but not crazy

    # -------------------------------------------------------------------------
    # Immediate move utility
    # -------------------------------------------------------------------------

    def _immediate_utility_of_move(
        self,
        move,
        board_state: board.Board,
        sensor_data: List[Tuple[bool, bool]],
    ) -> float:
        direction, move_type = self._unpack_move(move)
        if direction is None or move_type is None:
            return -1e9

        name = self._enum_name(move_type).lower()
        utility = 0.0

        # Base action rewards
        if "egg" in name:
            utility += 100.0
        elif "turd" in name:
            utility += 18.0  # slightly higher to encourage using turds
        else:
            utility += 0.5   # plain move is basically neutral

        # Destination-based features
        cur_x, cur_y = board_state.chicken_player.get_location()
        dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)

        # Bonus for landing on our parity (helps future egg chances)
        if self.my_egg_parity is not None:
            if (dest_x + dest_y) % 2 == self.my_egg_parity:
                utility += 4.0

        # Trapdoor risk: only care about *very* high risk
        risk = self._trapdoor_risk_at(dest_x, dest_y)
        high_risk_component = max(0.0, risk - 0.75)
        utility -= self.trapdoor_penalty * high_risk_component

        # Anti-camping penalties
        utility -= self._corner_camping_penalty(dest_x, dest_y)
        utility -= self._recent_position_penalty(dest_x, dest_y)

        # Don't sit on old egg squares forever
        if (dest_x, dest_y) in self.egg_squares:
            utility -= 5.0

        # Exploration bonus
        utility += self._distance_from_start_bonus(dest_x, dest_y)

        # Tiny noise to break ties
        utility += np.random.random() * 0.01

        return utility

    # -------------------------------------------------------------------------
    # Static board evaluation for minimax
    # -------------------------------------------------------------------------

    def _evaluate_board(self, board_state: board.Board) -> float:
        x, y = board_state.chicken_player.get_location()
        value = 0.0

        if self.my_egg_parity is not None:
            if (x + y) % 2 == self.my_egg_parity:
                value += 6.0

        value -= self._corner_camping_penalty(x, y)
        if (x, y) in self.egg_squares:
            value -= 4.0

        risk_here = self._trapdoor_risk_at(x, y)
        high_risk_component = max(0.0, risk_here - 0.75)
        value -= 0.5 * self.trapdoor_penalty * high_risk_component

        value += self._distance_from_start_bonus(x, y)

        return value

    def _minimax_leaf_value(self, board_state: Optional[board.Board]) -> float:
        if board_state is None:
            return 0.0
        return self._evaluate_board(board_state)

    # -------------------------------------------------------------------------
    # Alpha–beta minimax
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
        if board_state is None:
            return self._minimax_leaf_value(None)

        try:
            if time_left() < 0.4:
                return self._minimax_leaf_value(board_state)
        except Exception:
            pass

        moves = board_state.get_valid_moves()
        if depth == 0 or not moves:
            return self._minimax_leaf_value(board_state)

        scored_moves = [
            (self._immediate_utility_of_move(m, board_state, sensor_data), m)
            for m in moves
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=maximizing_player)

        if maximizing_player:
            value = -1e9
            expanded = False
            for _, move in scored_moves:
                direction, move_type = self._unpack_move(move)
                if direction is None or move_type is None:
                    continue
                try:
                    next_board = board_state.forecast_move(
                        direction, move_type)
                except (AttributeError, TypeError):
                    continue
                if next_board is None:
                    continue
                try:
                    next_board = next_board.reverse_perspective()
                except AttributeError:
                    pass

                expanded = True
                child = self._alpha_beta(
                    next_board, depth - 1, alpha, beta,
                    maximizing_player=False, time_left=time_left,
                    sensor_data=sensor_data,
                )
                if child > value:
                    value = child
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
            if not expanded:
                return self._minimax_leaf_value(board_state)
            return value
        else:
            value = 1e9
            expanded = False
            for _, move in scored_moves:
                direction, move_type = self._unpack_move(move)
                if direction is None or move_type is None:
                    continue
                try:
                    next_board = board_state.forecast_move(
                        direction, move_type)
                except (AttributeError, TypeError):
                    continue
                if next_board is None:
                    continue
                try:
                    next_board = next_board.reverse_perspective()
                except AttributeError:
                    pass

                expanded = True
                child = self._alpha_beta(
                    next_board, depth - 1, alpha, beta,
                    maximizing_player=True, time_left=time_left,
                    sensor_data=sensor_data,
                )
                if child < value:
                    value = child
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break
            if not expanded:
                return self._minimax_leaf_value(board_state)
            return value

    # -------------------------------------------------------------------------
    # Root move selection with iterative deepening
    # -------------------------------------------------------------------------

    def _choose_move_alpha_beta(
        self,
        board_state: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        moves = board_state.get_valid_moves()
        if not moves:
            return moves[0] if moves else None

        # 1. Greedy: always take an EGG if available.
        egg_moves = []
        for m in moves:
            _, mt = self._unpack_move(m)
            if mt is None:
                continue
            if "egg" in self._enum_name(mt).lower():
                egg_moves.append(m)
        if egg_moves:
            best = -1e9
            best_list: List[Tuple] = []
            for m in egg_moves:
                val = self._immediate_utility_of_move(
                    m, board_state, sensor_data)
                if val > best:
                    best = val
                    best_list = [m]
                elif val == best:
                    best_list.append(m)
            return best_list[np.random.randint(len(best_list))]

        # 2. Otherwise, use alpha–beta on remaining moves.

        base_scored = [
            (self._immediate_utility_of_move(m, board_state, sensor_data), m)
            for m in moves
        ]
        base_scored.sort(key=lambda x: x[0], reverse=True)

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

        best_move_overall = base_scored[0][1]
        best_value_overall = -1e9

        for depth in range(1, max_depth + 1):
            try:
                if time_left() < 0.8:
                    break
            except Exception:
                pass

            depth_best = -1e9
            depth_moves: List[Tuple] = []

            for _, move in base_scored:
                direction, move_type = self._unpack_move(move)
                if direction is None or move_type is None:
                    continue

                immediate = self._immediate_utility_of_move(
                    move, board_state, sensor_data
                )

                try:
                    next_board = board_state.forecast_move(
                        direction, move_type)
                except (AttributeError, TypeError):
                    total = immediate
                    if total > depth_best:
                        depth_best = total
                        depth_moves = [move]
                    elif total == depth_best:
                        depth_moves.append(move)
                    continue

                if next_board is None:
                    total = immediate
                    if total > depth_best:
                        depth_best = total
                        depth_moves = [move]
                    elif total == depth_best:
                        depth_moves.append(move)
                    continue

                try:
                    next_board = next_board.reverse_perspective()
                except AttributeError:
                    pass

                future = self._alpha_beta(
                    next_board, depth=depth - 1,
                    alpha=-1e9, beta=1e9,
                    maximizing_player=False,
                    time_left=time_left,
                    sensor_data=sensor_data,
                )
                total = immediate + self.gamma * future

                if total > depth_best:
                    depth_best = total
                    depth_moves = [move]
                elif total == depth_best:
                    depth_moves.append(move)

            if depth_moves:
                best_move_overall = depth_moves[np.random.randint(
                    len(depth_moves))]
                best_value_overall = depth_best

        return best_move_overall

    # -------------------------------------------------------------------------
    # Main entrypoint
    # -------------------------------------------------------------------------

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        self.turn_index += 1

        moves = board.get_valid_moves()
        if not moves:
            return moves[0] if moves else None

        loc = board.chicken_player.get_location()
        self._maybe_update_egg_parity(loc, moves)
        self._update_trapdoor_beliefs(loc, sensor_data)

        chosen_move = self._choose_move_alpha_beta(
            board, sensor_data, time_left)

        # Update anti-camping state based on where we go and if we lay an egg.
        direction, move_type = self._unpack_move(chosen_move)
        if direction is not None and move_type is not None:
            dest_x, dest_y = self._apply_direction(loc, direction)

            self.recent_positions.append((dest_x, dest_y))
            if len(self.recent_positions) > self.max_recent_positions:
                self.recent_positions.pop(0)

            if "egg" in self._enum_name(move_type).lower():
                self.egg_squares.add((dest_x, dest_y))

        return chosen_move
