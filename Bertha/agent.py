from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    STYLE B-PRIME: Improved Balanced Agent
    --------------------------------------
    - Early game: moderately aggressive, blocks opponent, uses turds + eggs tactically
    - Mid game: reduces aggression, increases map exploration
    - Late game: high safety, avoids trapdoors, prioritizes safe egg-laying
    - Smooth transitions based on turn count
    """

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.start_x, self.start_y = board.chicken_player.get_location()

        # Parity
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Movement memory
        self.visited_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 6
        self.prev_loc = None

        # Edge tracking
        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.last_edge_side = None

        # Egg memory
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Tuned parameters
        self.turn_index = 0
        self.gamma = 0.45     # future discount
        self.trapdoor_penalty = 0  # dynamic per phase — adjusted later

    # ------------------------------------------------------------
    # Trapdoor beliefs
    # ------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                # Simple even/odd belief
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 1
                else:
                    self.belief_black[y][x] = 1

        self.belief_white /= self.belief_white.sum()
        self.belief_black /= self.belief_black.sum()

    def _trapdoor_risk_at(self, x, y):
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return float(self.belief_white[y][x] + self.belief_black[y][x])
        return 0

    # ------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------
    def _enum(self, obj):
        return obj.name if hasattr(obj, "name") else str(obj)

    def _unpack(self, move):
        return move[0], move[1]

    def _apply_dir(self, loc, direction):
        x, y = loc
        d = self._enum(direction).lower()
        if "up" in d:
            return x, y - 1
        if "down" in d:
            return x, y + 1
        if "left" in d:
            return x - 1, y
        if "right" in d:
            return x + 1, y
        return x, y

    def _get_opponent_location(self, board_state):
        return board_state.chicken_enemy.get_location()

    def _edge_side(self, x, y):
        if y == 0:
            return 0
        if y == self.board_size - 1:
            return 1
        if x == 0:
            return 2
        if x == self.board_size - 1:
            return 3
        return None

    # ------------------------------------------------------------
    # Penalties
    # ------------------------------------------------------------
    def _visited_penalty(self, x, y):
        c = self.visited_counts.get((x, y), 0)
        return min(4, c - 1) if c > 1 else 0

    def _recent_penalty(self, x, y):
        p = 0
        for (px, py) in self.recent_positions:
            d = abs(px - x) + abs(py - y)
            if d == 0:
                p += 2
            elif d == 1:
                p += 1
        return p

    def _backtrack_penalty(self, dest):
        return 6.0 if dest == self.prev_loc else 0

    # ------------------------------------------------------------
    # Game-phase dependent weights
    # ------------------------------------------------------------
    def _phase(self):
        """
        Returns: "early", "mid", "late"
        """
        t = self.turn_index
        if t <= 12:
            return "early"
        elif t <= 24:
            return "mid"
        return "late"

    # ------------------------------------------------------------
    # Immediate utility
    # ------------------------------------------------------------
    def _immediate_utility(self, move, board_state, sensors):
        direction, move_type = self._unpack(move)
        name = self._enum(move_type).lower()

        cur_x, cur_y = board_state.chicken_player.get_location()
        dest_x, dest_y = self._apply_dir((cur_x, cur_y), direction)
        dest = (dest_x, dest_y)

        phase = self._phase()
        util = 0

        # ======================================================
        # 1. EGGS — balanced but phase-aware
        # ======================================================
        if "egg" in name:
            if phase == "early":
                util += 50  # early territory control
            elif phase == "mid":
                util += 80
            else:
                util += 110  # late-game egg priority

            if self.my_egg_parity and (dest_x + dest_y) % 2 == self.my_egg_parity:
                util += 4

        # ======================================================
        # 2. TURDS — offense early, defense mid, avoid late
        # ======================================================
        if "turd" in name:
            ox, oy = self._get_opponent_location(board_state)
            dist = abs(ox - dest_x) + abs(oy - dest_y)

            if phase == "early":
                # aggressive blocking
                if dist <= 2:
                    util += 25
                elif dist <= 3:
                    util += 10
                else:
                    util -= 6

                # trap opponent
                opp_moves = board_state.get_valid_moves(enemy=True)
                blocks = True
                for om in opp_moves:
                    od, ot = self._unpack(om)
                    ox2, oy2 = self._apply_dir((ox, oy), od)
                    if (ox2, oy2) != (dest_x, dest_y):
                        blocks = False
                        break
                if blocks:
                    util += 40

            elif phase == "mid":
                # more cautious
                if dist == 1:
                    util += 15
                elif dist <= 2:
                    util += 10
                else:
                    util -= 10

            else:
                # late game: avoid useless turds
                util -= 12

        # ======================================================
        # 3. Pursuit — strong early, weaker mid, avoided late
        # ======================================================
        ox, oy = self._get_opponent_location(board_state)
        before = abs(cur_x - ox) + abs(cur_y - oy)
        after = abs(dest_x - ox) + abs(dest_y - oy)

        if phase == "early":
            if after < before:
                util += 3
        elif phase == "mid":
            if after < before:
                util += 1.2
        else:
            # late game: avoid opponent
            if after < before:
                util -= 2
            else:
                util += 1

        # ======================================================
        # 4. Trapdoor penalty increases with time
        # ======================================================
        if phase == "early":
            trap_penalty = 40
        elif phase == "mid":
            trap_penalty = 80
        else:
            trap_penalty = 130  # medium-high late caution

        util -= trap_penalty * self._trapdoor_risk_at(dest_x, dest_y)

        # ======================================================
        # 5. Edge rotation logic — consistent
        # ======================================================
        side = self._edge_side(dest_x, dest_y)
        if side is not None:
            util += 1.3
            if self.edge_side_counts[side] == 0:
                util += 3
            if self.last_edge_side == side and self.edge_side_counts[side] > 8:
                util -= 1.2

        # ======================================================
        # 6. Anti-looping
        # ======================================================
        util -= self._visited_penalty(dest_x, dest_y)
        util -= self._recent_penalty(dest_x, dest_y)
        util -= self._backtrack_penalty(dest)

        if dest in self.egg_squares:
            util -= 2

        return util + np.random.random() * 0.01

    # ------------------------------------------------------------
    # Evaluation for minimax
    # ------------------------------------------------------------
    def _evaluate(self, board_state):
        x, y = board_state.chicken_player.get_location()
        v = - self._visited_penalty(x, y)
        v -= 80 * self._trapdoor_risk_at(x, y)
        return v

    # ------------------------------------------------------------
    # Alpha-Beta
    # ------------------------------------------------------------
    def _alpha_beta(self, board_state, depth, alpha, beta, maxing, time_left, sensors):
        if board_state is None:
            return -1e9

        try:
            if time_left() < 0.3:
                return self._evaluate(board_state)
        except:
            return self._evaluate(board_state)

        moves = board_state.get_valid_moves()
        if depth == 0 or not moves:
            return self._evaluate(board_state)

        if maxing:
            val = -1e9
            for m in moves:
                d, t = self._unpack(m)
                nb = board_state.forecast_move(d, t)
                if nb:
                    try:
                        nb = nb.reverse_perspective()
                    except:
                        pass
                    val = max(val, self._alpha_beta(nb, depth - 1, alpha, beta,
                                                    False, time_left, sensors))
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = 1e9
            for m in moves:
                d, t = self._unpack(m)
                nb = board_state.forecast_move(d, t)
                if nb:
                    try:
                        nb = nb.reverse_perspective()
                    except:
                        pass
                    val = min(val, self._alpha_beta(nb, depth - 1, alpha, beta,
                                                    True, time_left, sensors))
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val

    # ------------------------------------------------------------
    # Choose move
    # ------------------------------------------------------------
    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # Egg moves always immediate priority
        egg_moves = [m for m in moves if "egg" in self._enum(m[1]).lower()]
        if egg_moves:
            return max(egg_moves, key=lambda m: self._immediate_utility(m, board_state, sensors))

        # Score moves
        scored = [(self._immediate_utility(m, board_state, sensors), m)
                  for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Depth
        try:
            t = time_left()
        except:
            t = 999

        depth = 3 if t > 150 else 2
        best_m = scored[0][1]
        best_v = -1e9

        for _, m in scored:
            d, t2 = self._unpack(m)
            immediate = self._immediate_utility(m, board_state, sensors)
            nb = board_state.forecast_move(d, t2)

            if nb is None:
                total = immediate
            else:
                try:
                    nb = nb.reverse_perspective()
                except:
                    pass
                future = self._alpha_beta(
                    nb, depth - 1, -1e9, 1e9, False, time_left, sensors)
                total = immediate + self.gamma * future

            if total > best_v:
                best_v = total
                best_m = m

        return best_m

    # ------------------------------------------------------------
    # Main play
    # ------------------------------------------------------------
    def play(self, board_state, sensors, time_left):
        self.turn_index += 1

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        loc = board_state.chicken_player.get_location()

        # Detect parity
        if self.my_egg_parity is None:
            for m in moves:
                _, mt = self._unpack(m)
                if "egg" in self._enum(mt).lower():
                    x, y = loc
                    self.my_egg_parity = (x + y) % 2

        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            return moves[0]

        direction, move_type = self._unpack(chosen)
        dest_x, dest_y = self._apply_dir(loc, direction)
        dest = (dest_x, dest_y)

        # Update movement memory
        self.prev_loc = loc
        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1

        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        # Egg memory
        if "egg" in self._enum(move_type).lower():
            self.egg_squares.add(dest)

        # Edge memory
        side = self._edge_side(dest_x, dest_y)
        if side is not None:
            self.edge_side_counts[side] += 1
            self.last_edge_side = side

        return chosen
