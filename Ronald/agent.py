from collections.abc import Callable
from typing import List, Tuple, Optional, Set
import numpy as np

from game import *


"""
Exploring, trapdoor-aware, egg-greedy agent.

Key behavior changes vs previous version:
- Strongly encouraged to move away from its starting corner.
- Penalized for revisiting recent positions (no camping).
- Turds are discouraged early game, more valuable later.
- Still very egg-greedy and trapdoor-averse.
"""


class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = 8
        self.center_x = (self.board_size - 1) / 2
        self.center_y = (self.board_size - 1) / 2

        # Starting location (used for exploration bonus)
        self.start_x, self.start_y = board.chicken_player.get_location()

        # Egg parity (0 even, 1 odd) once we know if we're A or B.
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs
        self.belief_white = np.zeros((self.board_size, self.board_size), float)
        self.belief_black = np.zeros((self.board_size, self.board_size), float)
        self._init_trapdoor_beliefs()

        # Penalty multiplier for stepping on trapdoors
        self.trapdoor_penalty = 350.0

        # Future discount for alpha–beta
        self.gamma = 0.5

        # Anti-camping memory
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 8

        # Squares where we have already laid eggs
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Turn counter
        self.turn_index = 0

    # -------------------------------------------------------------------------
    # Trapdoor belief initialization
    # -------------------------------------------------------------------------

    def _init_trapdoor_beliefs(self):
        # White trapdoor: even squares
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 0:
                    self.belief_white[y, x] = 1.0
        self.belief_white /= self.belief_white.sum()

        # Black trapdoor: odd squares
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 1:
                    self.belief_black[y, x] = 1.0
        self.belief_black /= self.belief_black.sum()

    # -------------------------------------------------------------------------
    # General helpers
    # -------------------------------------------------------------------------

    def _enum_name(self, obj):
        return obj.name if hasattr(obj, "name") else str(obj)

    def _unpack_move(self, move):
        if not isinstance(move, tuple) or len(move) != 2:
            return None, None
        return move[0], move[1]

    def _apply_direction(self, loc: Tuple[int, int], direction) -> Tuple[int, int]:
        x, y = loc
        d = self._enum_name(direction).lower()
        if "up" in d:
            return x, y - 1
        if "down" in d:
            return x, y + 1
        if "left" in d:
            return x - 1, y
        if "right" in d:
            return x + 1, y
        return x, y

    def _trapdoor_risk_at(self, x: int, y: int) -> float:
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return 0.0
        return float(self.belief_white[y, x] + self.belief_black[y, x])

    # -------------------------------------------------------------------------
    # Sensor → belief update
    # -------------------------------------------------------------------------

    def _sensor_likelihoods(self, dx: int, dy: int) -> Tuple[float, float]:
        ax, ay = abs(dx), abs(dy)
        # Adjacent (sharing edge)
        if (ax == 1 and ay == 0) or (ax == 0 and ay == 1):
            return 0.5, 0.3
        # Diagonal
        if ax == 1 and ay == 1:
            return 0.25, 0.15
        # “Second ring”
        if max(ax, ay) == 2:
            return 0.10, 0.0
        return 0.0, 0.0

    def _update_single_belief(
        self,
        belief: np.ndarray,
        parity: int,
        loc: Tuple[int, int],
        heard: bool,
        felt: bool,
    ):
        x0, y0 = loc
        new_belief = np.zeros_like(belief)
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 != parity:
                    continue
                prior = belief[y, x]
                if prior <= 0.0:
                    continue
                dx, dy = x - x0, y - y0
                p_h, p_f = self._sensor_likelihoods(dx, dy)
                lh_hear = p_h if heard else 1.0 - p_h
                lh_feel = p_f if felt else 1.0 - p_f
                likelihood = lh_hear * lh_feel
                new_belief[y, x] = prior * likelihood

        total = new_belief.sum()
        if total > 1e-12:
            new_belief /= total
            belief[:, :] = new_belief

        # Standing on same-colour square and didn’t fall → that square not trapdoor
        if (x0 + y0) % 2 == parity:
            belief[y0, x0] = 0.0
            s = belief.sum()
            if s > 0:
                belief[:, :] = belief / s

    def _update_trapdoor_beliefs(
        self,
        loc: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]],
    ):
        (heard_w, felt_w) = sensor_data[0]
        (heard_b, felt_b) = sensor_data[1]
        self._update_single_belief(self.belief_white, 0, loc, heard_w, felt_w)
        self._update_single_belief(self.belief_black, 1, loc, heard_b, felt_b)

    # -------------------------------------------------------------------------
    # Parity (A vs B) detection
    # -------------------------------------------------------------------------

    def _maybe_set_parity(self, loc: Tuple[int, int], moves: List[Tuple]):
        if self.my_egg_parity is not None:
            return
        x, y = loc
        for m in moves:
            _, mt = self._unpack_move(m)
            if mt is None:
                continue
            if "egg" in self._enum_name(mt).lower():
                self.my_egg_parity = (x + y) % 2
                break

    # -------------------------------------------------------------------------
    # Penalties / bonuses for positions
    # -------------------------------------------------------------------------

    def _corner_penalty(self, x: int, y: int) -> float:
        d_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
        if d_edge == 0:
            return 3.0    # actual corner
        if d_edge == 1:
            return 1.0    # right next to corner
        return 0.0

    def _recent_penalty(self, x: int, y: int) -> float:
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
        Reward being far from the starting corner.
        This nudges the chicken to traverse the board/edges
        instead of sitting where it spawned.
        """
        d = abs(x - self.start_x) + abs(y - self.start_y)
        # Cap the bonus so it can't override trapdoor fear or egg priorities.
        return 0.4 * min(d, 8)

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
        util = 0.0

        # Base rewards:
        if "egg" in name:
            util += 100.0
        elif "turd" in name:
            # Turds are low-value early, decent later.
            base = 6.0
            if self.turn_index < 12:
                base -= 8.0  # strongly discourage early turds
            util += base
        else:
            util += 1.0  # plain move is modest but fine

        cur_x, cur_y = board_state.chicken_player.get_location()
        dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)

        # Egg­-parity bonus
        if self.my_egg_parity is not None:
            if (dest_x + dest_y) % 2 == self.my_egg_parity:
                util += 4.0

        # Trapdoor risk: strong aversion, especially if moving to riskier tile.
        risk_here = self._trapdoor_risk_at(cur_x, cur_y)
        risk_dest = self._trapdoor_risk_at(dest_x, dest_y)

        util -= self.trapdoor_penalty * risk_dest
        util -= 220.0 * max(0.0, risk_dest - risk_here)

        # Anti-camping:
        util -= self._corner_penalty(dest_x, dest_y)
        util -= self._recent_penalty(dest_x, dest_y)

        if (dest_x, dest_y) in self.egg_squares:
            util -= 5.0

        # Encouraged to roam away from starting corner / quadrant.
        util += self._distance_from_start_bonus(dest_x, dest_y)

        # Small noise to break ties
        util += np.random.random() * 0.01

        return util

    # -------------------------------------------------------------------------
    # Static board evaluation (for minimax leaves)
    # -------------------------------------------------------------------------

    def _evaluate_board(self, board_state: board.Board) -> float:
        x, y = board_state.chicken_player.get_location()
        value = 0.0

        if self.my_egg_parity is not None:
            if (x + y) % 2 == self.my_egg_parity:
                value += 6.0

        value -= self._corner_penalty(x, y)
        if (x, y) in self.egg_squares:
            value -= 4.0

        risk = self._trapdoor_risk_at(x, y)
        value -= self.trapdoor_penalty * risk

        value += self._distance_from_start_bonus(x, y)

        return value

    def _leaf_value(self, board_state: Optional[board.Board]) -> float:
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
        maximizing: bool,
        time_left: Callable,
        sensor_data: List[Tuple[bool, bool]],
    ) -> float:
        if board_state is None:
            return self._leaf_value(None)

        try:
            if time_left() < 0.3:
                return self._leaf_value(board_state)
        except Exception:
            pass

        moves = board_state.get_valid_moves()
        if depth == 0 or not moves:
            return self._leaf_value(board_state)

        scored = [
            (self._immediate_utility_of_move(m, board_state, sensor_data), m)
            for m in moves
        ]
        scored.sort(key=lambda x: x[0], reverse=maximizing)

        if maximizing:
            value = -1e9
            for _, move in scored:
                d, t = self._unpack_move(move)
                try:
                    nb = board_state.forecast_move(d, t)
                except Exception:
                    continue
                if nb is None:
                    continue
                try:
                    nb = nb.reverse_perspective()
                except AttributeError:
                    pass

                child = self._alpha_beta(
                    nb, depth - 1, alpha, beta, False, time_left, sensor_data
                )
                value = max(value, child)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = 1e9
            for _, move in scored:
                d, t = self._unpack_move(move)
                try:
                    nb = board_state.forecast_move(d, t)
                except Exception:
                    continue
                if nb is None:
                    continue
                try:
                    nb = nb.reverse_perspective()
                except AttributeError:
                    pass

                child = self._alpha_beta(
                    nb, depth - 1, alpha, beta, True, time_left, sensor_data
                )
                value = min(value, child)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # -------------------------------------------------------------------------
    # Root move selection: greedy eggs + alpha–beta
    # -------------------------------------------------------------------------

    def _choose_move(
        self,
        board_state: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # 1. Greedy: if Egg moves exist, choose the best one and skip search.
        egg_moves = []
        for m in moves:
            _, mt = self._unpack_move(m)
            if mt and "egg" in self._enum_name(mt).lower():
                egg_moves.append(m)

        if egg_moves:
            best_val = -1e9
            best_list: List[Tuple] = []
            for m in egg_moves:
                v = self._immediate_utility_of_move(
                    m, board_state, sensor_data)
                if v > best_val:
                    best_val = v
                    best_list = [m]
                elif v == best_val:
                    best_list.append(m)
            return best_list[np.random.randint(len(best_list))]

        # 2. No egg moves → alpha–beta with iterative deepening.
        base_scored = [
            (self._immediate_utility_of_move(m, board_state, sensor_data), m)
            for m in moves
        ]
        base_scored.sort(key=lambda x: x[0], reverse=True)

        try:
            remaining = time_left()
        except Exception:
            remaining = 9999

        if remaining > 180:
            max_depth = 4
        elif remaining > 60:
            max_depth = 3
        else:
            max_depth = 2

        best_move = base_scored[0][1]
        best_val_overall = -1e9

        for depth in range(1, max_depth + 1):
            try:
                if time_left() < 0.6:
                    break
            except Exception:
                pass

            depth_best_val = -1e9
            depth_best_moves: List[Tuple] = []

            for _, move in base_scored:
                d, t = self._unpack_move(move)
                immediate = self._immediate_utility_of_move(
                    move, board_state, sensor_data
                )

                try:
                    nb = board_state.forecast_move(d, t)
                except Exception:
                    nb = None

                if nb is None:
                    total = immediate
                else:
                    try:
                        nb = nb.reverse_perspective()
                    except AttributeError:
                        pass
                    future = self._alpha_beta(
                        nb,
                        depth=depth - 1,
                        alpha=-1e9,
                        beta=1e9,
                        maximizing=False,
                        time_left=time_left,
                        sensor_data=sensor_data,
                    )
                    total = immediate + self.gamma * future

                if total > depth_best_val:
                    depth_best_val = total
                    depth_best_moves = [move]
                elif total == depth_best_val:
                    depth_best_moves.append(move)

            if depth_best_moves:
                best_move = depth_best_moves[np.random.randint(
                    len(depth_best_moves))]
                best_val_overall = depth_best_val

        return best_move

    # -------------------------------------------------------------------------
    # Main entry point
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
            return None

        loc = board.chicken_player.get_location()
        self._maybe_set_parity(loc, moves)
        self._update_trapdoor_beliefs(loc, sensor_data)

        chosen = self._choose_move(board, sensor_data, time_left)
        if chosen is None:
            return moves[0]

        # Update anti-camping memory based on where we’re going / if we lay egg.
        direction, move_type = self._unpack_move(chosen)
        dest_x, dest_y = self._apply_direction(loc, direction)

        self.recent_positions.append((dest_x, dest_y))
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        if "egg" in self._enum_name(move_type).lower():
            self.egg_squares.add((dest_x, dest_y))

        return chosen
