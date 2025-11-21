from collections.abc import Callable
from typing import List, Tuple, Optional, Set
from collections import deque
from math import log, exp
import numpy as np

from game import *


class PlayerAgent:
    """
    Edge-walking, egg-greedy, trapdoor-aware agent with *mild* corner safety.

    Design:
    - Always prefers EGG moves when available.
    - Walks around edges instead of camping in one corner.
    - Uses sensor data to maintain trapdoor beliefs and avoid risky tiles.
    - Turds are discouraged early, used more mid/late game.
    - Corner logic is gentle: good if safe to escape, mildly bad if not.
    - Anti-camping via recent position penalties and avoiding old egg tiles.
    - Uses shallow alpha–beta minimax for non-egg turns.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = 8
        self.start_x, self.start_y = board.chicken_player.get_location()

        # Parity (0 even, 1 odd) – set once we see an Egg move
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs
        self.belief_white = np.zeros((self.board_size, self.board_size), float)
        self.belief_black = np.zeros((self.board_size, self.board_size), float)
        self._init_trapdoor_beliefs()

        # Penalty for stepping on trapdoor
        self.trapdoor_penalty = 320.0

        # Future discount in minimax
        self.gamma = 0.5

        # Anti-camping memory
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 8

        # Squares where we've laid eggs
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Turn index
        self.turn_index = 0

    # ----------------------------------------------------------------------
    # Belief init
    # ----------------------------------------------------------------------

    def _init_trapdoor_beliefs(self):
        # White trapdoor on even squares
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 0:
                    self.belief_white[y, x] = 1.0
        self.belief_white /= self.belief_white.sum()

        # Black trapdoor on odd squares
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 1:
                    self.belief_black[y, x] = 1.0
        self.belief_black /= self.belief_black.sum()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Sensor → belief update
    # ----------------------------------------------------------------------

    def _sensor_likelihoods(self, dx: int, dy: int) -> Tuple[float, float]:
        ax, ay = abs(dx), abs(dy)
        if (ax == 1 and ay == 0) or (ax == 0 and ay == 1):
            return 0.5, 0.3
        if ax == 1 and ay == 1:
            return 0.25, 0.15
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
                lh_hear = p_h if heard else (1.0 - p_h)
                lh_feel = p_f if felt else (1.0 - p_f)
                likelihood = lh_hear * lh_feel
                new_belief[y, x] = prior * likelihood

        total = new_belief.sum()
        if total > 1e-12:
            new_belief /= total
            belief[:, :] = new_belief

        # If we stood on that colour and did not fall, that square is safe
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

    # ----------------------------------------------------------------------
    # Parity detection (A vs B)
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Spatial penalties / bonuses
    # ----------------------------------------------------------------------

    def _corner_penalty(self, x: int, y: int) -> float:
        d_edge = min(x, y, self.board_size - 1 - x, self.board_size - 1 - y)
        if d_edge == 0:
            return 2.5     # actual corner
        if d_edge == 1:
            return 0.8     # right next to a corner
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

    def _edge_bonus(self, x: int, y: int) -> float:
        """
        Reward being on the outer ring but NOT in the exact corners.
        """
        on_left_or_right = (x == 0 or x == self.board_size - 1)
        on_top_or_bottom = (y == 0 or y == self.board_size - 1)
        is_corner = on_left_or_right and on_top_or_bottom

        if is_corner:
            return 0.2
        if on_left_or_right or on_top_or_bottom:
            return 2.0
        return 0.0

    def _center_penalty(self, x: int, y: int) -> float:
        # Very mild central penalty
        if 2 <= x <= 5 and 2 <= y <= 5:
            return 0.7
        return 0.0

    def _distance_from_start_bonus(self, x: int, y: int) -> float:
        d = abs(x - self.start_x) + abs(y - self.start_y)
        return 0.2 * min(d, 8)

    # ----------------------------------------------------------------------
    # Corner escape safety
    # ----------------------------------------------------------------------

    def _escape_safety_from(
        self,
        start: Tuple[int, int],
        max_depth: int = 4,
    ) -> float:
        """
        Approximate probability that there exists a reasonably safe path
        from `start` to some non-corner edge square, within `max_depth` steps.
        """
        sx, sy = start

        def neighbors(x: int, y: int):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    yield nx, ny

        visited_log = {}
        q = deque()
        q.append((sx, sy, 0, 0.0))
        visited_log[(sx, sy)] = 0.0

        best_safe_prob = 0.0

        while q:
            x, y, depth, log_safe = q.popleft()
            if depth >= max_depth:
                continue
            for nx, ny in neighbors(x, y):
                risk = self._trapdoor_risk_at(nx, ny)
                survive = max(1e-6, 1.0 - risk)
                new_log = log_safe + log(survive)

                if ((nx, ny) not in visited_log) or (new_log > visited_log[(nx, ny)]):
                    visited_log[(nx, ny)] = new_log
                    q.append((nx, ny, depth + 1, new_log))

                    on_edge = (nx == 0 or nx == self.board_size - 1 or
                               ny == 0 or ny == self.board_size - 1)
                    is_corner = (nx in (0, self.board_size - 1)
                                 and ny in (0, self.board_size - 1))
                    if on_edge and not is_corner:
                        safe_prob = exp(new_log)
                        if safe_prob > best_safe_prob:
                            best_safe_prob = safe_prob

        return float(max(0.0, min(1.0, best_safe_prob)))

    # ----------------------------------------------------------------------
    # Immediate move utility
    # ----------------------------------------------------------------------

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

        # Base rewards
        if "egg" in name:
            util += 100.0
        elif "turd" in name:
            # Turds: small negative early, better later
            if self.turn_index < 13:
                util -= 8.0
            elif self.turn_index < 23:
                util += 5.0
            else:
                util += 12.0
        else:
            util += 1.0

        cur_x, cur_y = board_state.chicken_player.get_location()
        dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)

        # Egg parity
        if self.my_egg_parity is not None:
            if (dest_x + dest_y) % 2 == self.my_egg_parity:
                util += 3.0

        # Trapdoor risk
        risk_here = self._trapdoor_risk_at(cur_x, cur_y)
        risk_dest = self._trapdoor_risk_at(dest_x, dest_y)
        util -= self.trapdoor_penalty * risk_dest
        util -= 150.0 * max(0.0, risk_dest - risk_here)

        # Spatial structure
        util += self._edge_bonus(dest_x, dest_y)
        util -= self._corner_penalty(dest_x, dest_y)
        util -= self._center_penalty(dest_x, dest_y)

        # Corner escape safety (gentle)
        is_corner = (
            dest_x in (0, self.board_size - 1)
            and dest_y in (0, self.board_size - 1)
        )
        if is_corner:
            corner_safety = self._escape_safety_from((dest_x, dest_y))
            if "egg" in name:
                util += 25.0 * corner_safety
            if corner_safety < 0.25:
                util -= 50.0 * (0.25 - corner_safety)

        # Anti-camping
        util -= self._recent_penalty(dest_x, dest_y)
        if (dest_x, dest_y) in self.egg_squares:
            util -= 4.0

        # Move away from spawn
        util += self._distance_from_start_bonus(dest_x, dest_y)

        # Tiny jitter
        util += np.random.random() * 0.01

        return util

    # ----------------------------------------------------------------------
    # Static evaluation
    # ----------------------------------------------------------------------

    def _evaluate_board(self, board_state: board.Board) -> float:
        x, y = board_state.chicken_player.get_location()
        value = 0.0

        if self.my_egg_parity is not None:
            if (x + y) % 2 == self.my_egg_parity:
                value += 5.0

        value += self._edge_bonus(x, y)
        value -= self._corner_penalty(x, y)
        value -= self._center_penalty(x, y)

        if (x, y) in self.egg_squares:
            value -= 3.0

        risk = self._trapdoor_risk_at(x, y)
        value -= self.trapdoor_penalty * risk

        value += self._distance_from_start_bonus(x, y)

        return value

    # ----------------------------------------------------------------------
    # Alpha–beta
    # ----------------------------------------------------------------------

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
            return 0.0

        try:
            if time_left() < 0.3:
                return self._evaluate_board(board_state)
        except Exception:
            pass

        moves = board_state.get_valid_moves()
        if depth == 0 or not moves:
            return self._evaluate_board(board_state)

        scored = [
            (self._immediate_utility_of_move(m, board_state, sensor_data), m)
            for m in moves
        ]
        scored.sort(key=lambda x: x[0], reverse=maximizing)

        if maximizing:
            value = -1e9
            for _, m in scored:
                d, t = self._unpack_move(m)
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
                v = self._alpha_beta(nb, depth - 1, alpha,
                                     beta, False, time_left, sensor_data)
                value = max(value, v)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = 1e9
            for _, m in scored:
                d, t = self._unpack_move(m)
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
                v = self._alpha_beta(nb, depth - 1, alpha,
                                     beta, True, time_left, sensor_data)
                value = min(value, v)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # ----------------------------------------------------------------------
    # Move selection
    # ----------------------------------------------------------------------

    def _choose_move(
        self,
        board_state: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # 1) Greedy EGG
        egg_moves = [
            m for m in moves
            if "egg" in self._enum_name(m[1]).lower()
        ]
        if egg_moves:
            best = -1e9
            best_list: List[Tuple] = []
            for m in egg_moves:
                v = self._immediate_utility_of_move(
                    m, board_state, sensor_data)
                if v > best:
                    best = v
                    best_list = [m]
                elif v == best:
                    best_list.append(m)
            return best_list[np.random.randint(len(best_list))]

        # 2) Otherwise alpha–beta
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
        best_value = -1e9

        for depth in range(1, max_depth + 1):
            try:
                if time_left() < 0.6:
                    break
            except Exception:
                pass

            depth_best = -1e9
            depth_moves: List[Tuple] = []

            for _, m in base_scored:
                d, t = self._unpack_move(m)
                immediate = self._immediate_utility_of_move(
                    m, board_state, sensor_data)

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
                        nb, depth - 1, -1e9, 1e9, False, time_left, sensor_data
                    )
                    total = immediate + self.gamma * future

                if total > depth_best:
                    depth_best = total
                    depth_moves = [m]
                elif total == depth_best:
                    depth_moves.append(m)

            if depth_moves:
                best_move = depth_moves[np.random.randint(len(depth_moves))]
                best_value = depth_best

        return best_move

    # ----------------------------------------------------------------------
    # Main entrypoint
    # ----------------------------------------------------------------------

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

        direction, move_type = self._unpack_move(chosen)
        dest_x, dest_y = self._apply_direction(loc, direction)

        self.recent_positions.append((dest_x, dest_y))
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        if "egg" in self._enum_name(move_type).lower():
            self.egg_squares.add((dest_x, dest_y))

        return chosen
