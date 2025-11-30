from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set, Iterable
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – PAMMY-SAFE (Zero Trap Risk Version)
# ================================================================
class PlayerAgent:
    """
    PAMMY-SAFE:
    - NEVER steps on trapdoors.
    - Hard safety thresholds + Bayesian updating on no-signal.
    - Predictive trap corridors; auto-blacklists rising danger zones.
    - Strong egg play retained.
    - Smart Turds T2-A retained but trap-safe.
    """

    # --------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------
    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE

        # Parity detection
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs (white + black parity)
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Danger memory
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()
        self.confirmed_danger_zone: Set[Tuple[int, int]] = set()
        self.prev_risk: Dict[Tuple[int, int], float] = {}

        # Exploration
        self.visited_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 10

        # Movement memory
        self.prev_loc: Optional[Tuple[int, int]] = None

        # Egg memory
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Edges
        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        self.turn_index = 0
        self.gamma = 0.22

        # Hard safety thresholds
        self.THRESH_EARLY = 0.025
        self.THRESH_MID = 0.018
        self.THRESH_LATE = 0.012

    # --------------------------------------------------------------
    # Trapdoor belief initialization
    # --------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                # White parity trap
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 1
                else:
                    self.belief_black[y][x] = 1
        self._normalize()

    def _normalize(self):
        self.belief_white /= self.belief_white.sum()
        self.belief_black /= self.belief_black.sum()

    # --------------------------------------------------------------
    # Trapdoor Danger
    # --------------------------------------------------------------
    def _trapdoor_risk(self, x, y):
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return float(self.belief_white[y][x] + self.belief_black[y][x])
        return 0.0

    def _update_prev_risk(self, x, y, new_risk):
        old = self.prev_risk.get((x, y), new_risk)
        self.prev_risk[(x, y)] = new_risk
        return new_risk > old + 0.01    # rising danger flag

    # --------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------
    def _apply_dir(self, loc: Tuple[int, int], d) -> Tuple[int, int]:
        x, y = loc
        name = d.name.lower()
        if "up" in name:
            return x, y - 1
        if "down" in name:
            return x, y + 1
        if "left" in name:
            return x - 1, y
        if "right" in name:
            return x + 1, y
        return x, y

    def _phase(self):
        if self.turn_index <= 14:
            return "early"
        if self.turn_index <= 26:
            return "mid"
        return "late"

    def _safety_threshold(self):
        ph = self._phase()
        if ph == "early":
            return self.THRESH_EARLY
        if ph == "mid":
            return self.THRESH_MID
        return self.THRESH_LATE

    def _extract_traps(self, board_state, sensors):
        known = set()
        try:
            known |= set(board_state.found_trapdoors)
        except:
            pass

        if isinstance(sensors, Iterable):
            for s in sensors:
                if isinstance(s, tuple) and len(s) == 2:
                    known.add(s)
        return known

    def _record_trap(self, loc):
        self.confirmed_trapdoors.add(loc)
        x, y = loc
        for (nx, ny) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                self.confirmed_danger_zone.add((nx, ny))

    # --------------------------------------------------------------
    # BAYESIAN UPDATE ON SENSORS
    # --------------------------------------------------------------
    def _bayes_update(self, loc, sensors):
        (hw, fw), (hb, fb) = sensors

        # Precompute proximity likelihood for all tiles
        for color, (heard, felt) in enumerate([(hw, fw), (hb, fb)]):

            grid = self.belief_white if color == 0 else self.belief_black
            like = np.ones_like(grid)

            for y in range(self.board_size):
                for x in range(self.board_size):
                    p_h, p_f = self._signal_prob(loc, (x, y))
                    like[y][x] *= (p_h if heard else (1 - p_h))
                    like[y][x] *= (p_f if felt else (1 - p_f))

            grid *= like

        self._normalize()

        # **NO SIGNAL UPDATE** → quiet tiles become safer
        if sensors == [(False, False), (False, False)]:
            lx, ly = loc
            for y in range(self.board_size):
                for x in range(self.board_size):
                    dist = abs(x - lx) + abs(y - ly)
                    if dist <= 1:
                        self.belief_white[y][x] *= 0.18
                        self.belief_black[y][x] *= 0.18
            self._normalize()

    def _signal_prob(self, here, trap):
        hx, hy = here
        tx, ty = trap
        dx, dy = abs(hx - tx), abs(hy - ty)

        # direct adjacency
        if dx + dy == 1:
            return 0.50, 0.30
        if dx == 1 and dy == 1:
            return 0.25, 0.15
        if dx + dy == 2:
            return 0.10, 0.00
        return 0.00, 0.00

    # --------------------------------------------------------------
    # Rising Danger Corridor Detection
    # --------------------------------------------------------------
    def _is_rising_danger(self, x, y):
        r = self._trapdoor_risk(x, y)
        return self._update_prev_risk(x, y, r)

    # --------------------------------------------------------------
    # Immediate Utility
    # --------------------------------------------------------------
    def _immediate_utility(self, move, board_state, sensors):
        d, mt = move
        cx, cy = board_state.chicken_player.get_location()
        nx, ny = self._apply_dir((cx, cy), d)
        dest = (nx, ny)

        risk = self._trapdoor_risk(nx, ny)
        th = self._safety_threshold()

        # ABSOLUTE BAN on high-risk squares
        if risk > th:
            return -1e12

        # Rising danger corridor → banned
        if self._is_rising_danger(nx, ny):
            return -5e11

        util = 0.0
        t = self.turn_index

        # exploration
        if dest not in self.visited_counts:
            util += 150
        else:
            util -= 38 * self.visited_counts[dest]

        # avoid stepping back
        if dest == self.prev_loc:
            util -= 40

        # avoid repeated loops
        util -= 15 * self.recent_positions.count(dest)

        # egg scoring
        if mt == MoveType.EGG:
            if risk < th * 0.8:
                util += 120
            elif risk < th * 0.5:
                util += 60
            if self.my_egg_parity is not None:
                if ((nx + ny) % 2) == self.my_egg_parity:
                    util += 6

        # turd scoring
        if mt == MoveType.TURD:
            ox, oy = board_state.chicken_enemy.get_location()
            dist = abs(nx - ox) + abs(ny - oy)

            if dist <= 4 and risk < th * 0.7:
                util += 25
                if nx == ox or ny == oy:
                    util += 18
            else:
                util -= 25

        return util + np.random.random() * 0.002

    # --------------------------------------------------------------
    # MINIMAX WRAPPER
    # --------------------------------------------------------------
    def _mini(self, state, move, sensors, time_left):
        d, mt = move
        nxt = state.forecast_move(d, mt)
        if nxt is None:
            return -1e9

        # Opponent turn evaluation
        if hasattr(nxt, "reverse_perspective"):
            nxt.reverse_perspective()

        moves2 = nxt.get_valid_moves()
        if not moves2:
            return self._immediate_utility(move, state, sensors)

        worst = 1e9
        for m2 in moves2:
            d2, mt2 = m2
            nxt2 = nxt.forecast_move(d2, mt2)
            if nxt2 is None:
                continue
            val = self._immediate_utility(move, state, sensors)
            worst = min(worst, val)
        return worst

    # --------------------------------------------------------------
    # MOVE SELECTION (with trap-safe filtering)
    # --------------------------------------------------------------
    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()
        th = self._safety_threshold()

        # HARD SAFETY FILTER — remove trapdanger moves
        safe_moves = []
        for m in moves:
            d, mt = m
            nx, ny = self._apply_dir((cx, cy), d)
            r = self._trapdoor_risk(nx, ny)
            if r <= th and (nx, ny) not in self.confirmed_danger_zone:
                safe_moves.append(m)

        if safe_moves:
            moves = safe_moves
        else:
            # extreme fallback: choose lowest-risk move
            moves = sorted(moves, key=lambda m: self._trapdoor_risk(
                *self._apply_dir((cx, cy), m[0])))

        # Score moves
        scored = [(self._immediate_utility(m, board_state, sensors), m)
                  for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        # take best
        return scored[0][1]

    # --------------------------------------------------------------
    # MAIN API
    # --------------------------------------------------------------
    def play(self, board_state: board.Board, sensors, time_left: Callable):

        self.turn_index += 1

        # update traps
        known = self._extract_traps(board_state, sensors)
        for loc in known:
            if loc not in self.confirmed_trapdoors:
                self._record_trap(loc)

        # BAYESIAN SIGNAL UPDATE
        cx, cy = board_state.chicken_player.get_location()
        self._bayes_update((cx, cy), sensors)

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # detect egg parity
        if self.my_egg_parity is None:
            for m in moves:
                if m[1] == MoveType.EGG:
                    self.my_egg_parity = (cx + cy) % 2

        # choose
        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            return moves[0]

        # record movement
        d, mt = chosen
        nx, ny = self._apply_dir((cx, cy), d)

        self.prev_loc = (cx, cy)
        dest = (nx, ny)

        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1
        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        if mt == MoveType.EGG:
            self.egg_squares.add(dest)

        return chosen
