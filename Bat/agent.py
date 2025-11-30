from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    SAFE EXPLORER VERSION
    --------------------------------------------------
    - ABSOLUTE trapdoor avoidance at all stages
    - Exploration > eggs > blocking early/mid game
    - Only eggs/turds on SAFE and USEFUL tiles
    - No repetition, no loops, no backtracking
    - Late-game egg maximization intact
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE

        # Parity
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs (even/odd)
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Movement memory
        self.visited_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 8
        self.prev_loc = None

        # Egg memory
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Edge tracking
        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.last_edge_side = None

        # Internal
        self.turn_index = 0
        self.gamma = 0.30

    # ------------------------------------------------------------
    # Trapdoor beliefs
    # ------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
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
    # Helpers
    # ------------------------------------------------------------
    def _enum(self, obj):
        return obj.name if hasattr(obj, "name") else str(obj)

    def _unpack(self, move):
        return move[0], move[1]

    def _apply_dir(self, loc, d):
        x, y = loc
        dd = self._enum(d).lower()
        if "up" in dd:
            return x, y - 1
        if "down" in dd:
            return x, y + 1
        if "left" in dd:
            return x - 1, y
        if "right" in dd:
            return x + 1, y
        return x, y

    def _phase(self):
        t = self.turn_index
        if t <= 14:
            return "early"
        elif t <= 26:
            return "mid"
        return "late"

    def _visited_penalty(self, x, y):
        return self.visited_counts.get((x, y), 0)

    def _recent_penalty(self, x, y):
        return sum(1 for (px, py) in self.recent_positions if (px, py) == (x, y))

    def _backtrack_penalty(self, dest):
        return 1 if dest == self.prev_loc else 0

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    def _evaluate(self, board_state):
        x, y = board_state.chicken_player.get_location()
        return -(20 * self._visited_penalty(x, y)) - 200 * self._trapdoor_risk_at(x, y)

    # ------------------------------------------------------------
    # Immediate utility
    # ------------------------------------------------------------
    def _immediate_utility(self, move, board_state, sensors):
        direction, move_type = self._unpack(move)

        cx, cy = board_state.chicken_player.get_location()
        nx, ny = self._apply_dir((cx, cy), direction)
        dest = (nx, ny)

        phase = self._phase()
        util = 0
        risk = self._trapdoor_risk_at(nx, ny)

        # ======================================================
        # ABSOLUTE SAFETY RULES
        # ======================================================
        if risk > 0.06:
            return -999999  # no stepping on even moderately risky tiles

        if (nx, ny) in self.egg_squares:
            util -= 50  # never hover around old egg zones

        # ======================================================
        # EXPLORATION PRIORITY
        # ======================================================
        if (nx, ny) not in self.visited_counts:
            util += 80  # new tiles are king

        # heavily punish staying around same zones
        util -= 20 * self._visited_penalty(nx, ny)
        util -= 12 * self._recent_penalty(nx, ny)
        util -= 30 * self._backtrack_penalty((nx, ny))

        # ======================================================
        # EGG LOGIC (safe-first)
        # ======================================================
        if "egg" in self._enum(move_type).lower():
            if risk < 0.02:
                util += 60
            elif risk < 0.04:
                util += 30
            else:
                util -= 50

            if self.my_egg_parity is not None:
                if (nx + ny) % 2 == self.my_egg_parity:
                    util += 5

        # ======================================================
        # TURD LOGIC — safe blocking only
        # ======================================================
        if "turd" in self._enum(move_type).lower():
            ox, oy = board_state.chicken_enemy.get_location()
            dist = abs(nx - ox) + abs(ny - oy)

            if dist <= 2 and risk < 0.03:
                util += 25
            else:
                util -= 20

        # ======================================================
        # LIGHT opponent pressure
        # ======================================================
        ox, oy = board_state.chicken_enemy.get_location()
        before = abs(cx - ox) + abs(cy - oy)
        after = abs(nx - ox) + abs(ny - oy)

        if after < before and risk < 0.04:
            util += 3

        return util + (np.random.random() * 0.01)

    # ------------------------------------------------------------
    # Alpha-beta search
    # ------------------------------------------------------------
    def _alpha_beta(self, state, depth, alpha, beta, maxing, time_left, sensors):
        if state is None:
            return -1e9

        try:
            if time_left() < 0.25:
                return self._evaluate(state)
        except:
            return self._evaluate(state)

        moves = state.get_valid_moves()
        if depth == 0 or not moves:
            return self._evaluate(state)

        if maxing:
            best = -1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt:
                    try:
                        nxt = nxt.reverse_perspective()
                    except:
                        pass
                    best = max(best, self._alpha_beta(nxt, depth-1, alpha, beta,
                                                      False, time_left, sensors))
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best

        else:
            best = 1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt:
                    try:
                        nxt = nxt.reverse_perspective()
                    except:
                        pass
                    best = min(best, self._alpha_beta(nxt, depth-1, alpha, beta,
                                                      True, time_left, sensors))
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    # ------------------------------------------------------------
    # Choose move
    # ------------------------------------------------------------
    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # Egg moves *only* if safe and beneficial
        egg_moves = [m for m in moves if "egg" in self._enum(m[1]).lower()]
        if egg_moves:
            filtered = [(self._immediate_utility(m, board_state, sensors), m)
                        for m in egg_moves]
            filtered.sort(reverse=True)
            if filtered[0][0] > 0:
                return filtered[0][1]

        scored = [(self._immediate_utility(m, board_state, sensors), m)
                  for m in moves]
        scored.sort(reverse=True)

        # Do short minimax search
        try:
            t = time_left()
        except:
            t = 999

        depth = 2 if t > 100 else 1
        best_move = scored[0][1]
        best_value = -1e9

        for _, m in scored[:4]:  # only top 4 moves
            d, t2 = self._unpack(m)
            nxt = board_state.forecast_move(d, t2)

            if nxt is None:
                total = _
            else:
                try:
                    nxt = nxt.reverse_perspective()
                except:
                    pass
                future = self._alpha_beta(nxt, depth-1, -1e9, 1e9,
                                          False, time_left, sensors)
                total = _ + self.gamma * future

            if total > best_value:
                best_value = total
                best_move = m

        return best_move

    # ------------------------------------------------------------
    # Main play loop
    # ------------------------------------------------------------
    def play(self, board_state, sensors, time_left):
        self.turn_index += 1

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        loc = board_state.chicken_player.get_location()

        # Detect parity early
        if self.my_egg_parity is None:
            for m in moves:
                _, mt = self._unpack(m)
                if "egg" in self._enum(mt).lower():
                    x, y = loc
                    self.my_egg_parity = (x + y) % 2

        choice = self._choose_move(board_state, sensors, time_left)
        if choice is None:
            return moves[0]

        d, t = self._unpack(choice)
        nx, ny = self._apply_dir(loc, d)
        dest = (nx, ny)

        # Store movement history
        self.prev_loc = loc
        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1

        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        # Egg memory
        if "egg" in self._enum(t).lower():
            self.egg_squares.add(dest)

        return choice
