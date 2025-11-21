from collections.abc import Callable
from typing import List, Tuple, Optional, Set
import numpy as np

from game import *


"""
SAFE MINIM / ULYSSES AGENT
-----------------------------------------
Goals:
1. Lay as many eggs as possible.
2. Avoid trapdoors aggressively.
3. Do NOT camp or freeze in corners.
4. Do NOT wander into high-risk tiles.
5. Use iterative deepening alpha–beta search.
6. Update trapdoor beliefs correctly using sensor data.

This agent prioritizes survival + egg farming.
"""


class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = 8
        self.center_x = (self.board_size - 1) / 2
        self.center_y = (self.board_size - 1) / 2

        # Parity (0 = even, 1 = odd)
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs
        self.belief_white = np.zeros((self.board_size, self.board_size), float)
        self.belief_black = np.zeros((self.board_size, self.board_size), float)
        self._init_trapdoor_beliefs()

        # Very strong deterrent
        self.trapdoor_penalty = 400.0

        # Alpha–beta discount for future value
        self.gamma = 0.50

        # Memory to discourage camping
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 6

        # Squares where we already laid eggs (don’t go back)
        self.egg_squares: Set[Tuple[int, int]] = set()

        self.turn_index = 0

        # record starting location for reference
        self.start_x, self.start_y = board.chicken_player.get_location()

    # ------------------------------------------------------------
    # Belief Initialization
    # ------------------------------------------------------------

    def _init_trapdoor_beliefs(self):
        # White: even squares
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 1
        self.belief_white /= self.belief_white.sum()

        # Black: odd squares
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 == 1:
                    self.belief_black[y][x] = 1
        self.belief_black /= self.belief_black.sum()

    # ------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------

    def _enum_name(self, obj):
        return obj.name if hasattr(obj, "name") else str(obj)

    def _unpack_move(self, move):
        if not isinstance(move, tuple) or len(move) != 2:
            return None, None
        return move[0], move[1]

    def _apply_direction(self, loc, direction):
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

    def _trapdoor_risk_at(self, x, y):
        if not (0 <= x < 8 and 0 <= y < 8):
            return 0
        return float(self.belief_white[y][x] + self.belief_black[y][x])

    # ------------------------------------------------------------
    # Sensor → Belief Update
    # ------------------------------------------------------------

    def _sensor_likelihoods(self, dx, dy):
        ax, ay = abs(dx), abs(dy)
        if (ax == 1 and ay == 0) or (ax == 0 and ay == 1):
            return 0.5, 0.3        # edge neighbor
        if ax == 1 and ay == 1:
            return 0.25, 0.15     # diagonal
        if max(ax, ay) == 2:
            return 0.1, 0.0       # second ring
        return 0.0, 0.0

    def _update_single_belief(self, belief, parity, loc, heard, felt):
        x0, y0 = loc
        new_belief = np.zeros_like(belief)
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 != parity:
                    continue
                prior = belief[y][x]
                if prior <= 0:
                    continue
                dx, dy = x - x0, y - y0
                p_h, p_f = self._sensor_likelihoods(dx, dy)
                lh_hear = p_h if heard else 1 - p_h
                lh_feel = p_f if felt else 1 - p_f
                likelihood = lh_hear * lh_feel
                new_belief[y][x] = prior * likelihood

        total = new_belief.sum()
        if total > 1e-9:
            new_belief /= total
            belief[:] = new_belief

        # If we're standing on that color, it can't be a trapdoor
        if (x0 + y0) % 2 == parity:
            belief[y0][x0] = 0
            s = belief.sum()
            if s > 0:
                belief[:] = belief / s

    def _update_trapdoor_beliefs(self, loc, sensor):
        (hw, fw) = sensor[0]
        (hb, fb) = sensor[1]
        self._update_single_belief(self.belief_white, 0, loc, hw, fw)
        self._update_single_belief(self.belief_black, 1, loc, hb, fb)

    # ------------------------------------------------------------
    # Parity detection (A or B)
    # ------------------------------------------------------------

    def _maybe_set_parity(self, loc, moves):
        if self.my_egg_parity is not None:
            return
        x, y = loc
        for m in moves:
            _, mt = self._unpack_move(m)
            if mt and "egg" in self._enum_name(mt).lower():
                self.my_egg_parity = (x + y) % 2
                break

    # ------------------------------------------------------------
    # Penalties & Bonuses
    # ------------------------------------------------------------

    def _corner_penalty(self, x, y):
        d_edge = min(x, y, 7 - x, 7 - y)
        if d_edge == 0:
            return 3.0
        if d_edge == 1:
            return 1.5
        return 0.0

    def _recent_penalty(self, x, y):
        penalty = 0
        for px, py in self.recent_positions:
            d = abs(px - x) + abs(py - y)
            if d == 0:
                penalty += 4
            elif d == 1:
                penalty += 2
        return penalty

    # ------------------------------------------------------------
    # Immediate move utility (MAIN SAFE LOGIC)
    # ------------------------------------------------------------

    def _immediate_utility_of_move(self, move, board_state, sensor_data):
        direction, move_type = self._unpack_move(move)
        if direction is None:
            return -1e9

        name = self._enum_name(move_type).lower()
        utility = 0

        # Base rewards
        if "egg" in name:
            utility += 100
        elif "turd" in name:
            utility += 18
        else:
            utility += 0.3

        cur_x, cur_y = board_state.chicken_player.get_location()
        dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)

        # Correct parity bonus
        if self.my_egg_parity is not None:
            if (dest_x + dest_y) % 2 == self.my_egg_parity:
                utility += 4

        # Trapdoor risk penalties
        risk_here = self._trapdoor_risk_at(cur_x, cur_y)
        risk_dest = self._trapdoor_risk_at(dest_x, dest_y)

        utility -= self.trapdoor_penalty * risk_dest
        utility -= 250.0 * max(0, risk_dest - risk_here)

        # Anti-camping
        utility -= self._corner_penalty(dest_x, dest_y)
        utility -= self._recent_penalty(dest_x, dest_y)

        if (dest_x, dest_y) in self.egg_squares:
            utility -= 5.0

        # tiny jitter
        utility += np.random.random() * 0.01

        return utility

    # ------------------------------------------------------------
    # Board evaluation for minimax
    # ------------------------------------------------------------

    def _evaluate_board(self, board_state):
        x, y = board_state.chicken_player.get_location()
        value = 0

        # correct parity bonus
        if self.my_egg_parity is not None:
            if (x + y) % 2 == self.my_egg_parity:
                value += 6

        # camping
        value -= self._corner_penalty(x, y)
        if (x, y) in self.egg_squares:
            value -= 4

        # risk
        risk = self._trapdoor_risk_at(x, y)
        value -= self.trapdoor_penalty * risk

        return value

    # ------------------------------------------------------------
    # Alpha–Beta search
    # ------------------------------------------------------------

    def _alpha_beta(self, board_state, depth, alpha, beta, maximizing, time_left, sensor):
        if board_state is None:
            return 0

        try:
            if time_left() < 0.3:
                return self._evaluate_board(board_state)
        except:
            pass

        moves = board_state.get_valid_moves()
        if depth == 0 or not moves:
            return self._evaluate_board(board_state)

        ordered = [
            (self._immediate_utility_of_move(m, board_state, sensor), m)
            for m in moves
        ]
        ordered.sort(key=lambda x: x[0], reverse=maximizing)

        if maximizing:
            value = -1e9
            for _, move in ordered:
                d, t = self._unpack_move(move)
                try:
                    nb = board_state.forecast_move(d, t)
                except:
                    continue
                if nb is None:
                    continue
                try:
                    nb = nb.reverse_perspective()
                except:
                    pass

                v = self._alpha_beta(
                    nb, depth - 1, alpha, beta, False, time_left, sensor
                )
                value = max(value, v)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        else:
            value = 1e9
            for _, move in ordered:
                d, t = self._unpack_move(move)
                try:
                    nb = board_state.forecast_move(d, t)
                except:
                    continue
                if nb is None:
                    continue
                try:
                    nb = nb.reverse_perspective()
                except:
                    pass

                v = self._alpha_beta(
                    nb, depth - 1, alpha, beta, True, time_left, sensor
                )
                value = min(value, v)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # ------------------------------------------------------------
    # Move selection (Egg-first + alpha–beta)
    # ------------------------------------------------------------

    def _choose_move(self, board_state, sensor, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # 1. Greedy egg move
        egg_moves = [
            m for m in moves
            if "egg" in self._enum_name(m[1]).lower()
        ]
        if egg_moves:
            best = -1e9
            best_list = []
            for m in egg_moves:
                v = self._immediate_utility_of_move(m, board_state, sensor)
                if v > best:
                    best = v
                    best_list = [m]
                elif v == best:
                    best_list.append(m)
            return best_list[np.random.randint(len(best_list))]

        # 2. No egg → alpha–beta search
        base_scores = [
            (self._immediate_utility_of_move(m, board_state, sensor), m)
            for m in moves
        ]
        base_scores.sort(key=lambda x: x[0], reverse=True)

        try:
            t = time_left()
        except:
            t = 9999

        if t > 180:
            max_depth = 4
        elif t > 60:
            max_depth = 3
        else:
            max_depth = 2

        best_move = base_scores[0][1]
        best_val = -1e9

        for depth in range(1, max_depth + 1):
            try:
                if time_left() < 0.5:
                    break
            except:
                pass

            current_best = -1e9
            current_moves = []

            for _, move in base_scores:
                d, mt = self._unpack_move(move)
                immediate = self._immediate_utility_of_move(
                    move, board_state, sensor)

                try:
                    nb = board_state.forecast_move(d, mt)
                except:
                    nb = None
                if nb is None:
                    total = immediate
                    if total > current_best:
                        current_best = total
                        current_moves = [move]
                    elif total == current_best:
                        current_moves.append(move)
                    continue

                try:
                    nb = nb.reverse_perspective()
                except:
                    pass

                future = self._alpha_beta(
                    nb, depth - 1, -1e9, 1e9, False, time_left, sensor
                )
                total = immediate + self.gamma * future

                if total > current_best:
                    current_best = total
                    current_moves = [move]
                elif total == current_best:
                    current_moves.append(move)

            if current_moves:
                best_move = current_moves[np.random.randint(
                    len(current_moves))]
                best_val = current_best

        return best_move

    # ------------------------------------------------------------
    # Main play function
    # ------------------------------------------------------------

    def play(self, board, sensor_data, time_left):
        self.turn_index += 1

        loc = board.chicken_player.get_location()
        moves = board.get_valid_moves()
        if not moves:
            return None

        self._maybe_set_parity(loc, moves)
        self._update_trapdoor_beliefs(loc, sensor_data)

        chosen = self._choose_move(board, sensor_data, time_left)

        if chosen:
            direction, move_type = self._unpack_move(chosen)
            dest = self._apply_direction(loc, direction)

            # update memory
            self.recent_positions.append(dest)
            if len(self.recent_positions) > self.max_recent_positions:
                self.recent_positions.pop(0)

            # record egg tile so we avoid it later
            if "egg" in self._enum_name(move_type).lower():
                self.egg_squares.add(dest)

        return chosen
