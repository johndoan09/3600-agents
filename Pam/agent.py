from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set, Iterable
import numpy as np

from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    STYLE A + SMART TURDS (BOOSTED)
    - Avoids trapdoors
    - Strong exploration
    - Lays eggs only on safe tiles
    - TURDS NOW MORE FREQUENT (but still smart)
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE

        self.my_egg_parity = None

        # Trapdoor belief
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Trap memory
        self.confirmed_trapdoors = set()
        self.confirmed_danger_zone = set()

        # Exploration memory
        self.visited_counts = {}
        self.recent_positions = []
        self.max_recent_positions = 10
        self.prev_loc = None
        self.momentum_dir = None

        self.egg_squares = set()

        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.last_edge_side = None

        self.turn_index = 0
        self.gamma = 0.22

    # ------------------------------------------------------------
    # Trapdoor belief
    # ------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x+y) % 2 == 0:
                    self.belief_white[y][x] = 1
                else:
                    self.belief_black[y][x] = 1
        self.belief_white /= self.belief_white.sum()
        self.belief_black /= self.belief_black.sum()

    def _trapdoor_risk_at(self, x, y):
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return float(self.belief_white[y][x]+self.belief_black[y][x])
        return 0.0

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _enum(self, obj):
        return obj.name if hasattr(obj, "name") else str(obj)

    def _unpack(self, move):
        return move[0], move[1]

    def _apply_dir(self, loc, d):
        x, y = loc
        s = self._enum(d).lower()
        if "up" in s:
            return x, y-1
        if "down" in s:
            return x, y+1
        if "left" in s:
            return x-1, y
        if "right" in s:
            return x+1, y
        return x, y

    def _phase(self):
        if self.turn_index <= 14:
            return "early"
        if self.turn_index <= 26:
            return "mid"
        return "late"

    def _visited_penalty(self, x, y):
        return self.visited_counts.get((x, y), 0)

    def _recent_penalty(self, x, y):
        return sum(1 for (px, py) in self.recent_positions if (px, py) == (x, y))

    def _backtrack_penalty(self, dest):
        return 1 if dest == self.prev_loc else 0

    def _edge_side(self, x, y):
        if y == 0:
            return 0
        if y == self.board_size-1:
            return 1
        if x == 0:
            return 2
        if x == self.board_size-1:
            return 3
        return None

    def _get_opponent_location(self, b):
        return b.chicken_enemy.get_location()

    # ------------------------------------------------------------
    # Trap extraction
    # ------------------------------------------------------------
    def _extract_known_traps(self, b, sensors):
        known = set()
        try:
            known |= set(b.found_trapdoors)
        except:
            pass
        if isinstance(sensors, Iterable):
            for s in sensors:
                if isinstance(s, tuple) and len(s) == 2:
                    known.add(s)
        return known

    def _record_trapdoor_location(self, loc):
        self.confirmed_trapdoors.add(loc)
        x, y = loc
        for (nx, ny) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                self.confirmed_danger_zone.add((nx, ny))

    # ------------------------------------------------------------
    # Predictive turd scoring (BOOSTED)
    # ------------------------------------------------------------
    def _predictive_turd_value(self, nx, ny, ox, oy):
        base = 60  # BOOSTED from 25 → increases turd frequency

        corridor = 0
        if nx == ox or ny == oy:
            corridor = 25  # BOOSTED from 20

        dist = abs(nx-ox)+abs(ny-oy)
        return base + corridor - dist*4   # reduce distance penalty

    # ------------------------------------------------------------
    # Evaluation for minimax
    # ------------------------------------------------------------
    def _evaluate(self, b):
        x, y = b.chicken_player.get_location()
        return -70*self._visited_penalty(x, y) - 300*self._trapdoor_risk_at(x, y)

    # ------------------------------------------------------------
    # Immediate utility
    # ------------------------------------------------------------
    def _immediate_utility(self, move, b, sensors):
        d, mt = self._unpack(move)
        name = self._enum(mt).lower()

        cx, cy = b.chicken_player.get_location()
        nx, ny = self._apply_dir((cx, cy), d)
        dest = (nx, ny)

        util = 0
        phase = self._phase()
        risk = self._trapdoor_risk_at(nx, ny)

        # trap hard punish
        known = self._extract_known_traps(b, sensors)
        if dest in known or dest in self.confirmed_trapdoors:
            return -1e12
        if dest in self.confirmed_danger_zone:
            util -= 900
        util -= 200*risk

        # exploration
        if (nx, ny) not in self.visited_counts:
            util += 150
        else:
            util -= 40*self._visited_penalty(nx, ny)
            util -= 20*self._recent_penalty(nx, ny)

        util -= 50*self._backtrack_penalty(dest)
        if dest in self.egg_squares:
            util -= 40

        # opponent
        ox, oy = self._get_opponent_location(b)
        dist = abs(nx-ox)+abs(ny-oy)

        # eggs
        if "egg" in name:
            if risk < 0.02:
                util += 130
            elif risk < 0.04:
                util += 70
            else:
                util -= 80
            if self.my_egg_parity and ((nx+ny) % 2) == self.my_egg_parity:
                util += 4

        # --------------------------------------------------------
        # TURDS (SMART + BOOSTED FREQUENCY)
        # --------------------------------------------------------
        if "turd" in name:
            # booster 1: allow dropping on visited tiles now
            safe = (risk < 0.05)

            if dist <= 5 and safe:
                # booster 2: heavily increase blocking value
                block_score = self._predictive_turd_value(nx, ny, ox, oy)
                util += block_score   # used to be much smaller
            else:
                util -= 15  # mild penalty instead of harsh one

        # edges
        side = self._edge_side(nx, ny)
        if side is not None:
            util += 5
            if self.edge_side_counts[side] > 12:
                util -= 5

        return util + np.random.random()*0.002

    # ------------------------------------------------------------
    # Minimax
    # ------------------------------------------------------------
    def _alpha_beta(self, state, depth, a, b, maximizing, time_left, sensors):
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

        if maximizing:
            v = -1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt:
                    nxt.reverse_perspective()
                    v = max(v, self._alpha_beta(nxt, depth-1,
                            a, b, False, time_left, sensors))
                a = max(a, v)
                if a >= b:
                    break
            return v
        else:
            v = 1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt:
                    nxt.reverse_perspective()
                    v = min(v, self._alpha_beta(nxt, depth-1,
                            a, b, True, time_left, sensors))
                b = min(b, v)
                if a >= b:
                    break
            return v

    # ------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------
    def _choose_move(self, b, sensors, time_left):
        moves = b.get_valid_moves()
        if not moves:
            return None

        scored = [(self._immediate_utility(m, b, sensors), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        # safe egg priority
        eggs = [m for (sc, m) in scored if "egg" in self._enum(
            m[1]).lower() and sc > 0]
        if eggs:
            return eggs[0]

        # minimax
        try:
            tl = time_left()
        except:
            tl = 999
        depth = 2 if tl > 70 else 1

        best_move = scored[0][1]
        best_score = -1e9

        top_k = min(5, len(scored))
        for i in range(top_k):
            imm, m = scored[i]
            d, t = self._unpack(m)
            nxt = b.forecast_move(d, t)
            if nxt:
                nxt.reverse_perspective()
                fut = self._alpha_beta(
                    nxt, depth-1, -1e9, 1e9, False, time_left, sensors)
                total = imm+self.gamma*fut
            else:
                total = imm
            if total > best_score:
                best_score = total
                best_move = m
        return best_move

    # ------------------------------------------------------------
    # Main play
    # ------------------------------------------------------------
    def play(self, b, sensors, time_left):
        self.turn_index += 1

        # trap update
        known = self._extract_known_traps(b, sensors)
        for loc in known:
            if loc not in self.confirmed_trapdoors:
                self._record_trapdoor_location(loc)

        moves = b.get_valid_moves()
        if not moves:
            return None

        x0, y0 = b.chicken_player.get_location()

        if self.my_egg_parity is None:
            for mv in moves:
                _, mt = self._unpack(mv)
                if "egg" in self._enum(mt).lower():
                    self.my_egg_parity = (x0+y0) % 2
                    break

        chosen = self._choose_move(b, sensors, time_left)
        if chosen is None:
            chosen = moves[0]

        d, mt = self._unpack(chosen)
        nx, ny = self._apply_dir((x0, y0), d)
        dest = (nx, ny)

        self.prev_loc = (x0, y0)
        self.visited_counts[dest] = self.visited_counts.get(dest, 0)+1

        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        if "egg" in self._enum(mt).lower():
            self.egg_squares.add(dest)

        side = self._edge_side(nx, ny)
        if side is not None:
            self.edge_side_counts[side] += 1
            self.last_edge_side = side

        return chosen
