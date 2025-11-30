from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set, Iterable
import numpy as np

from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    SAFE EXPLORER + TRAPDOOR-AWARE + TURDS

    Priorities:
    1) Avoid known and sampled trapdoors as much as possible.
    2) Strongly prefer unvisited, unexplored tiles.
    3) Lay eggs on safe, new tiles (especially later in the game).
    4) Use turds when they safely block the opponent and are on new tiles.
    5) Avoid looping and backtracking.
    """

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE

        # Parity (for egg tiles)
        self.my_egg_parity: Optional[int] = None

        # Simple even/odd trapdoor belief (fallback heuristic)
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Movement memory
        self.visited_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 10
        self.prev_loc: Optional[Tuple[int, int]] = None

        # Egg memory
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Edge tracking
        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.last_edge_side: Optional[int] = None

        # Control
        self.turn_index = 0
        self.gamma = 0.3  # how much we care about minimax future

    # ------------------------------------------------------------
    # Trapdoor belief setup
    # ------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        # Simple assumption: trapdoors on both parities, uniform.
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 1
                else:
                    self.belief_black[y][x] = 1

        if self.belief_white.sum() > 0:
            self.belief_white /= self.belief_white.sum()
        if self.belief_black.sum() > 0:
            self.belief_black /= self.belief_black.sum()

    def _trapdoor_risk_at(self, x: int, y: int) -> float:
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return float(self.belief_white[y][x] + self.belief_black[y][x])
        return 0.0

    # ------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------
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

    def _visited_penalty(self, x: int, y: int) -> int:
        return self.visited_counts.get((x, y), 0)

    def _recent_penalty(self, x: int, y: int) -> int:
        return sum(1 for (px, py) in self.recent_positions if (px, py) == (x, y))

    def _backtrack_penalty(self, dest: Tuple[int, int]) -> int:
        return 1 if self.prev_loc is not None and dest == self.prev_loc else 0

    def _edge_side(self, x: int, y: int) -> Optional[int]:
        if y == 0:
            return 0
        if y == self.board_size - 1:
            return 1
        if x == 0:
            return 2
        if x == self.board_size - 1:
            return 3
        return None

    def _get_opponent_location(self, board_state: board.Board) -> Tuple[int, int]:
        return board_state.chicken_enemy.get_location()

    # Known trapdoors from board + sensors
    def _get_known_trapdoors(
        self,
        board_state: board.Board,
        sensors,
    ) -> Set[Tuple[int, int]]:
        known: Set[Tuple[int, int]] = set()

        # Found trapdoors in Board (if engine populates this)
        if hasattr(board_state, "found_trapdoors"):
            try:
                known |= set(board_state.found_trapdoors)
            except TypeError:
                pass

        # Sensor samples likely contain trapdoor coordinates
        if isinstance(sensors, Iterable):
            for s in sensors:
                if isinstance(s, tuple) and len(s) == 2:
                    known.add(s)

        return known

    # ------------------------------------------------------------
    # Evaluation function (for minimax)
    # ------------------------------------------------------------
    def _evaluate(self, board_state: board.Board) -> float:
        x, y = board_state.chicken_player.get_location()
        # prefer not being on often-visited tiles and low-risk tiles
        return (
            -30 * self._visited_penalty(x, y)
            - 150 * self._trapdoor_risk_at(x, y)
        )

    # ------------------------------------------------------------
    # Immediate utility (core behavior)
    # ------------------------------------------------------------
    def _immediate_utility(self, move, board_state: board.Board, sensors) -> float:
        direction, move_type = self._unpack(move)
        name = self._enum(move_type).lower()

        cx, cy = board_state.chicken_player.get_location()
        nx, ny = self._apply_dir((cx, cy), direction)
        dest = (nx, ny)

        phase = self._phase()
        util = 0.0

        # Known trapdoors from engine
        known_traps = self._get_known_trapdoors(board_state, sensors)

        # =======================================================
        # HARD TRAPDOOR AVOIDANCE
        # =======================================================
        if dest in known_traps:
            return -1e9  # absolutely never step on a known trapdoor

        # Also avoid immediate neighbors of known traps a bit
        for tx, ty in known_traps:
            if abs(tx - nx) + abs(ty - ny) == 1:
                util -= 200

        # Soft heuristic risk (even/odd belief)
        risk = self._trapdoor_risk_at(nx, ny)

        # Don't rely only on belief, but still penalize higher-risk tiles
        util -= 200 * risk

        # =======================================================
        # EXPLORATION PRIORITY
        # =======================================================
        if (nx, ny) not in self.visited_counts:
            # Exploration is very valuable to you
            util += 80
        else:
            # Heavily penalize revisits
            util -= 25 * self._visited_penalty(nx, ny)
            util -= 15 * self._recent_penalty(nx, ny)

        # Strongly discourage backtracking
        util -= 40 * self._backtrack_penalty(dest)

        # Avoid hovering where we already have eggs
        if dest in self.egg_squares:
            util -= 60

        # =======================================================
        # EGG LOGIC
        # =======================================================
        if "egg" in name:
            # Only lay eggs on tiles that are safe-ish
            if dest in known_traps:
                util -= 500
            elif risk < 0.02:
                util += 60
            elif risk < 0.04:
                util += 30
            else:
                util -= 40

            # Parity bonus
            if self.my_egg_parity is not None:
                if (nx + ny) % 2 == self.my_egg_parity:
                    util += 5

        # =======================================================
        # TURD LOGIC — SAFE BLOCKING
        # =======================================================
        if "turd" in name:
            ox, oy = self._get_opponent_location(board_state)
            dist = abs(nx - ox) + abs(ny - oy)

            # Prefer turds that are near the opponent *and* on new tiles
            if dist <= 2 and (nx, ny) not in self.visited_counts and risk < 0.03:
                util += 30  # now it should actually lay turds
            else:
                util -= 10  # discourage pointless turds

        # =======================================================
        # LIGHT OPPONENT PRESSURE (but not overriding safety)
        # =======================================================
        ox, oy = self._get_opponent_location(board_state)
        before = abs(cx - ox) + abs(cy - oy)
        after = abs(nx - ox) + abs(ny - oy)

        # Only very mild pursuit, and only if tile is new
        if after < before and (nx, ny) not in self.visited_counts and risk < 0.04:
            util += 5

        return util + np.random.random() * 0.01

    # ------------------------------------------------------------
    # Alpha–beta search (fixed reverse_perspective usage)
    # ------------------------------------------------------------
    def _alpha_beta(
        self,
        state: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        time_left: Callable,
        sensors,
    ) -> float:
        if state is None:
            return -1e9

        try:
            if time_left() < 0.25:
                return self._evaluate(state)
        except Exception:
            return self._evaluate(state)

        moves = state.get_valid_moves()
        if depth == 0 or not moves:
            return self._evaluate(state)

        if maximizing:
            best = -1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt is not None:
                    # Switch perspective to see from opponent's POV
                    if hasattr(nxt, "reverse_perspective"):
                        nxt.reverse_perspective()
                    best = max(
                        best,
                        self._alpha_beta(
                            nxt, depth - 1, alpha, beta, False, time_left, sensors
                        ),
                    )
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = 1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt is not None:
                    if hasattr(nxt, "reverse_perspective"):
                        nxt.reverse_perspective()
                    best = min(
                        best,
                        self._alpha_beta(
                            nxt, depth - 1, alpha, beta, True, time_left, sensors
                        ),
                    )
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    # ------------------------------------------------------------
    # Move selection with exploration + minimax
    # ------------------------------------------------------------
    def _choose_move(self, board_state: board.Board, sensors, time_left: Callable):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # First, score all moves by immediate utility
        scored = [(self._immediate_utility(m, board_state, sensors), m)
                  for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Try to lay an egg if it's a good egg move
        egg_candidates = [
            (score, m)
            for (score, m) in scored
            if "egg" in self._enum(m[1]).lower()
        ]
        if egg_candidates and egg_candidates[0][0] > 0:
            return egg_candidates[0][1]

        # Time-based depth
        try:
            tleft = time_left()
        except Exception:
            tleft = 999
        depth = 2 if tleft > 100 else 1

        best_move = scored[0][1]
        best_value = -1e9

        # Only run minimax on the top few candidates to save time
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
                future = self._alpha_beta(
                    nxt, depth - 1, -1e9, 1e9, False, time_left, sensors
                )
                total = imm + self.gamma * future

            if total > best_value:
                best_value = total
                best_move = m

        return best_move

    # ------------------------------------------------------------
    # Main play function
    # ------------------------------------------------------------
    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        loc = board_state.chicken_player.get_location()

        # Detect parity for egg parity bonus
        if self.my_egg_parity is None:
            for m in moves:
                _, mt = self._unpack(m)
                if "egg" in self._enum(mt).lower():
                    x, y = loc
                    self.my_egg_parity = (x + y) % 2
                    break

        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            chosen = moves[0]

        direction, move_type = self._unpack(chosen)
        nx, ny = self._apply_dir(loc, direction)
        dest = (nx, ny)

        # Update movement memory
        self.prev_loc = loc
        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1

        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        # Track egg placements
        if "egg" in self._enum(move_type).lower():
            self.egg_squares.add(dest)

        # Track edge usage
        side = self._edge_side(nx, ny)
        if side is not None:
            self.edge_side_counts[side] += 1
            self.last_edge_side = side

        return chosen
