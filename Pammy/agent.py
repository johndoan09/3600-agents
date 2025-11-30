from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set, Iterable
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – STYLE A + SMART TURDS T2-A
# ================================================================
class PlayerAgent:
    """
    STYLE A (Hyper-Safe Explorer)
    + SMART TURDS T2-A (Predictive blocking, not reactive)

    GOALS:
    - Avoid all trapdoors and danger corridors.
    - Strong avoid revisiting tiles to maximize egg potential.
    - Explore edges aggressively, interior only when safe.
    - Lay eggs only on safe, unvisited tiles.
    - Lay turds ONLY when they strategically block opponent routes.
    """

    # --------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------
    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE

        # Parity tracking
        self.my_egg_parity: Optional[int] = None

        # Trapdoor belief
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Permanent trap storage
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()
        self.confirmed_danger_zone: Set[Tuple[int, int]] = set()

        # Exploration memory
        self.visited_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 10
        self.prev_loc: Optional[Tuple[int, int]] = None
        self.momentum_dir: Optional[Direction] = None

        # Egg positions
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Edges
        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.last_edge_side: Optional[int] = None

        # Turn + minimax
        self.turn_index = 0
        self.gamma = 0.22   # future discount

    # --------------------------------------------------------------
    # Trapdoor belief initialization
    # --------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 1
                else:
                    self.belief_black[y][x] = 1

        self.belief_white /= self.belief_white.sum()
        self.belief_black /= self.belief_black.sum()

    def _trapdoor_risk_at(self, x: int, y: int) -> float:
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return float(self.belief_white[y][x] + self.belief_black[y][x])
        return 0.0

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _enum(self, obj):
        return obj.name if hasattr(obj, "name") else str(obj)

    def _unpack(self, move):
        return move[0], move[1]

    def _apply_dir(self, loc: Tuple[int, int], d) -> Tuple[int, int]:
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

    def _phase(self) -> str:
        t = self.turn_index
        if t <= 14:
            return "early"
        elif t <= 26:
            return "mid"
        return "late"

    # revisit penalty
    def _visited_penalty(self, x, y):
        return self.visited_counts.get((x, y), 0)

    # immediate repeat penalty
    def _recent_penalty(self, x, y):
        return sum(1 for (px, py) in self.recent_positions if (px, py) == (x, y))

    # no stepping right back
    def _backtrack_penalty(self, dest):
        return 1 if dest == self.prev_loc else 0

    # edge side classifier
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

    def _get_opponent_location(self, board_state):
        return board_state.chicken_enemy.get_location()

    # --------------------------------------------------------------
    # Trap detection
    # --------------------------------------------------------------
    def _extract_known_traps(self, board_state: board.Board, sensors):
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

    def _record_trapdoor_location(self, loc):
        self.confirmed_trapdoors.add(loc)
        x, y = loc
        for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                self.confirmed_danger_zone.add((nx, ny))

    # --------------------------------------------------------------
    # Minimax evaluation
    # --------------------------------------------------------------
    def _evaluate(self, board_state: board.Board) -> float:
        x, y = board_state.chicken_player.get_location()
        return (
            -70 * self._visited_penalty(x, y)
            - 300 * self._trapdoor_risk_at(x, y)
        )

    # --------------------------------------------------------------
    # Predictive turd target scoring (T2-A)
    # --------------------------------------------------------------
    def _predictive_turd_value(self, nx, ny, ox, oy):
        """
        Predict whether this tile blocks meaningful opponent routes.
        Logic:
        - Compute Manhattan shells around opponent.
        - If (nx,ny) lies on a shortest-path toward open space / eggs, give strong bonus.
        """

        dist_now = abs(ox - nx) + abs(oy - ny)

        # if tile lies in same row/column corridor → high block value
        corridor_bonus = 0
        if nx == ox or ny == oy:
            corridor_bonus = 20

        return 25 + corridor_bonus - dist_now

    # --------------------------------------------------------------
    # Immediate utility function
    # --------------------------------------------------------------
    def _immediate_utility(self, move, board_state: board.Board, sensors):
        d, mt = self._unpack(move)
        name = self._enum(mt).lower()

        cx, cy = board_state.chicken_player.get_location()
        nx, ny = self._apply_dir((cx, cy), d)
        dest = (nx, ny)
        phase = self._phase()
        util = 0.0

        # trap detection
        known = self._extract_known_traps(board_state, sensors)
        if dest in known or dest in self.confirmed_trapdoors:
            return -1e12
        if dest in self.confirmed_danger_zone:
            util -= 900

        # soft risk
        risk = self._trapdoor_risk_at(nx, ny)
        util -= 200 * risk

        # early game center-penalty (Style A)
        if phase == "early" and (3 <= nx <= 4 or 3 <= ny <= 4):
            util -= 60

        # exploration reward
        if (nx, ny) not in self.visited_counts:
            util += 150
        else:
            util -= 40 * self._visited_penalty(nx, ny)
            util -= 20 * self._recent_penalty(nx, ny)

        util -= 50 * self._backtrack_penalty(dest)

        if dest in self.egg_squares:
            util -= 40

        # opponent info
        ox, oy = self._get_opponent_location(board_state)

        # eggs
        if "egg" in name:
            if risk < 0.02:
                util += 130
            elif risk < 0.04:
                util += 70
            else:
                util -= 80

            if self.my_egg_parity and ((nx + ny) % 2) == self.my_egg_parity:
                util += 4

        # SMART TURDS T2-A
        if "turd" in name:
            dist = abs(nx - ox) + abs(ny - oy)

            if dist > 4:
                util -= 30
            else:
                if risk < 0.03 and (nx, ny) not in self.visited_counts:
                    blocking_score = self._predictive_turd_value(
                        nx, ny, ox, oy)
                    util += blocking_score
                else:
                    util -= 20

        # edges are good
        side = self._edge_side(nx, ny)
        if side is not None:
            util += 5
            if self.edge_side_counts[side] > 12:
                util -= 5

        return util + np.random.random() * 0.002

    # --------------------------------------------------------------
    # Minimax
    # --------------------------------------------------------------
    def _alpha_beta(self, state, depth, alpha, beta, maximizing, time_left, sensors):
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
            value = -1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt:
                    if hasattr(nxt, "reverse_perspective"):
                        nxt.reverse_perspective()
                    value = max(value, self._alpha_beta(
                        nxt, depth - 1, alpha, beta, False, time_left, sensors))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        else:
            value = 1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt:
                    if hasattr(nxt, "reverse_perspective"):
                        nxt.reverse_perspective()
                    value = min(value, self._alpha_beta(
                        nxt, depth - 1, alpha, beta, True, time_left, sensors))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # --------------------------------------------------------------
    # Choose move
    # --------------------------------------------------------------
    def _choose_move(self, board_state: board.Board, sensors, time_left: Callable):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        scored = [(self._immediate_utility(m, board_state, sensors), m)
                  for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        # prefer safe egg move
        egg_moves = [m for (sc, m) in scored if "egg" in self._enum(
            m[1]).lower() and sc > 0]
        if egg_moves:
            return egg_moves[0]

        # minimax for top moves
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
            nxt = board_state.forecast_move(d, t)
            if nxt is None:
                total = imm
            else:
                if hasattr(nxt, "reverse_perspective"):
                    nxt.reverse_perspective()
                fut = self._alpha_beta(
                    nxt, depth - 1, -1e9, 1e9, False, time_left, sensors)
                total = imm + self.gamma * fut

            if total > best_score:
                best_score = total
                best_move = m

        return best_move

    # --------------------------------------------------------------
    # Main play fn
    # --------------------------------------------------------------
    def play(self, board_state: board.Board, sensors, time_left: Callable):

        self.turn_index += 1

        # record traps
        known = self._extract_known_traps(board_state, sensors)
        for loc in known:
            if loc not in self.confirmed_trapdoors:
                self._record_trapdoor_location(loc)

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        loc = board_state.chicken_player.get_location()
        x0, y0 = loc

        if self.my_egg_parity is None:
            for m in moves:
                _, mt = self._unpack(m)
                if "egg" in self._enum(mt).lower():
                    self.my_egg_parity = (x0 + y0) % 2
                    break

        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            return moves[0]

        d, mt = self._unpack(chosen)
        nx, ny = self._apply_dir(loc, d)
        dest = (nx, ny)

        # record movement
        self.prev_loc = loc
        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1

        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        # record egg
        if "egg" in self._enum(mt).lower():
            self.egg_squares.add(dest)

        # record edge side
        side = self._edge_side(nx, ny)
        if side is not None:
            self.edge_side_counts[side] += 1
            self.last_edge_side = side

        return chosen
