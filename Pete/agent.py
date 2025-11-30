from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set, Iterable
import numpy as np

from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    BALANCED AGENT (Option B) + SMART TURDS + 2-STEP OPPONENT PREDICTION
    ---------------------------------------------------------------------
    - Avoids trapdoors & danger zones (Balanced B)
    - Strong exploration & unvisited tile preference
    - Drops turds when opponent is near or threatening egg zones
    - Drops turds to block reachable & future-reachable tiles
    - 2-step opponent movement & egg-parity prediction
    - Safe egg placement
    - Minimax tactical lookahead
    """

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def __init__(self, board: board.Board, time_left: Callable):

        self.board_size = board.game_map.MAP_SIZE

        # Parity for egg-laying
        self.my_egg_parity: Optional[int] = None

        # Belief model fallback (even/odd trapdoor heuristics)
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Trapdoor memory
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()
        self.confirmed_danger_zone: Set[Tuple[int, int]] = set()

        # Movement memory
        self.visited_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 10
        self.prev_loc: Optional[Tuple[int, int]] = None

        # Egg memory
        self.egg_squares: Set[Tuple[int, int]] = set()

        # Edge navigation
        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.last_edge_side: Optional[int] = None

        # Turn counter
        self.turn_index = 0
        self.gamma = 0.30  # minimax future discount

    # ----------------------------------------------------------------------
    # Trapdoor belief model
    # ----------------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
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

    # ----------------------------------------------------------------------
    # Basic helpers
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Trapdoor extraction and memory
    # ----------------------------------------------------------------------
    def _extract_known_traps(self, board_state: board.Board, sensors) -> Set[Tuple[int, int]]:
        known: Set[Tuple[int, int]] = set()

        # From board
        if hasattr(board_state, "found_trapdoors"):
            try:
                known |= set(board_state.found_trapdoors)
            except Exception:
                pass

        # From sensors (trapdoor samples)
        if isinstance(sensors, Iterable):
            for s in sensors:
                if isinstance(s, tuple) and len(s) == 2:
                    known.add(s)

        return known

    def _record_trapdoor_location(self, loc: Tuple[int, int]):
        # Hard record trap tile
        self.confirmed_trapdoors.add(loc)

        # Record danger corridor (neighbors)
        x, y = loc
        neighbors = [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1)
        ]
        for nx, ny in neighbors:
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                self.confirmed_danger_zone.add((nx, ny))
    # ----------------------------------------------------------------------
    # Opponent reachable tiles — 1-step and 2-step prediction
    # ----------------------------------------------------------------------

    def _opponent_reachable_1step(self, ox: int, oy: int) -> Set[Tuple[int, int]]:
        result = set()
        for d in Direction:
            nx, ny = self._apply_dir((ox, oy), d)
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                result.add((nx, ny))
        return result

    def _opponent_reachable_2step(self, ox: int, oy: int) -> Set[Tuple[int, int]]:
        """Where opponent can be in 2 moves."""
        s1 = self._opponent_reachable_1step(ox, oy)
        result = set(s1)
        for (x, y) in s1:
            result |= self._opponent_reachable_1step(x, y)
        return result

    # ----------------------------------------------------------------------
    # Board evaluation for minimax fallback
    # ----------------------------------------------------------------------
    def _evaluate(self, board_state: board.Board) -> float:
        x, y = board_state.chicken_player.get_location()
        return (
            -40 * self._visited_penalty(x, y)
            - 200 * self._trapdoor_risk_at(x, y)
        )

    # ----------------------------------------------------------------------
    # Immediate Utility — Main scoring function
    # ----------------------------------------------------------------------
    def _immediate_utility(self, move, board_state: board.Board, sensors) -> float:
        direction, move_type = self._unpack(move)
        name = self._enum(move_type).lower()

        cx, cy = board_state.chicken_player.get_location()
        nx, ny = self._apply_dir((cx, cy), direction)
        dest = (nx, ny)

        phase = self._phase()
        util = 0.0

        # ---------------------------------------------------------------
        # Trapdoor extraction and safety checks
        # ---------------------------------------------------------------
        known_traps = self._extract_known_traps(board_state, sensors)

        # If stepping on known trap
        if dest in known_traps or dest in self.confirmed_trapdoors:
            return -1e12

        # Strong penalty for trap corridor
        if dest in self.confirmed_danger_zone:
            util -= 900

        # Soft trap risk
        risk = self._trapdoor_risk_at(nx, ny)
        util -= 150 * risk

        # ---------------------------------------------------------------
        # Avoid center early (balanced Option B)
        # ---------------------------------------------------------------
        if phase == "early" and (3 <= nx <= 4 or 3 <= ny <= 4):
            util -= 60

        # ---------------------------------------------------------------
        # Exploration scoring
        # ---------------------------------------------------------------
        if (nx, ny) not in self.visited_counts:
            util += 120
        else:
            util -= 30 * self._visited_penalty(nx, ny)
            util -= 15 * self._recent_penalty(nx, ny)

        # Anti-backtracking
        util -= 40 * self._backtrack_penalty(dest)

        # Avoid stepping on own egg tiles
        if dest in self.egg_squares:
            util -= 40

        # ---------------------------------------------------------------
        # Opponent position
        # ---------------------------------------------------------------
        ox, oy = self._get_opponent_location(board_state)
        before = abs(cx - ox) + abs(cy - oy)
        after = abs(nx - ox) + abs(ny - oy)

        # Mild pursuit if tile is safe
        if risk < 0.04 and (nx, ny) not in self.visited_counts:
            if after < before:
                util += 5

        # ---------------------------------------------------------------
        # Egg logic — safer eggs with parity
        # ---------------------------------------------------------------
        if "egg" in name:

            # Only lay eggs on very safe tiles
            if risk < 0.02:
                util += 90
            elif risk < 0.04:
                util += 40
            else:
                util -= 60

            # Parity encouragement
            if self.my_egg_parity is not None:
                if (nx + ny) % 2 == self.my_egg_parity:
                    util += 4

        # ---------------------------------------------------------------
        # SMART TURD LOGIC — Balanced + 2-Step Opponent Prediction
        # ---------------------------------------------------------------
        if "turd" in name:

            # Dist to opponent
            dist = abs(nx - ox) + abs(ny - oy)

            # Opp egg parity
            op_parity = (ox + oy) % 2

            # Opponent next-turn reachable
            op_reach_1 = self._opponent_reachable_1step(ox, oy)

            # Opponent 2-step reachable
            op_reach_2 = self._opponent_reachable_2step(ox, oy)

            # Opponent parity-reachable (egg threat)
            op_parity_reach = {
                p for p in op_reach_1 | op_reach_2
                if (p[0] + p[1]) % 2 == op_parity
            }

            # -----------------------------------------------------------
            # Hard safety: never turd in danger areas
            # -----------------------------------------------------------
            if dest in self.confirmed_trapdoors or dest in self.confirmed_danger_zone:
                util -= 999
            elif risk > 0.06:
                util -= 200

            # -----------------------------------------------------------
            # 1. Strong turd bonus for being close to opponent
            # -----------------------------------------------------------
            if dist == 1:
                util += 180
            elif dist == 2:
                util += 120
            elif dist == 3:
                util += 40

            # -----------------------------------------------------------
            # 2. Block opponent egg-laying opportunities
            # -----------------------------------------------------------
            if (nx + ny) % 2 == op_parity:
                util += 110

            if dest in op_parity_reach:
                util += 90

            # -----------------------------------------------------------
            # 3. Block reachable future tiles (1-step & 2-step)
            # -----------------------------------------------------------
            if dest in op_reach_1:
                util += 70
            if dest in op_reach_2:
                util += 40

            # -----------------------------------------------------------
            # 4. Bonus for dropping turds on high-value unvisited tiles
            # -----------------------------------------------------------
            if (nx, ny) not in self.visited_counts:
                util += 55
            else:
                util -= 15

            # -----------------------------------------------------------
            # 5. Protect own eggs
            # -----------------------------------------------------------
            if any(abs(nx - ex) + abs(ny - ey) <= 1 for (ex, ey) in self.egg_squares):
                util += 100

            # -----------------------------------------------------------
            # 6. Penalty for wasteful far-away turds
            # -----------------------------------------------------------
            if dist > 4:
                util -= 45

        # ---------------------------------------------------------------
        # Edge navigation
        # ---------------------------------------------------------------
        side = self._edge_side(nx, ny)
        if side is not None:
            util += 5
            if self.edge_side_counts[side] > 12:
                util -= 4

        # Tiny noise to prevent deterministic ties
        return util + np.random.random() * 0.0005
    # ----------------------------------------------------------------------
    # Alpha-Beta Minimax with time-awareness
    # ----------------------------------------------------------------------

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

        # Abort search if low on time
        try:
            if time_left() < 0.25:
                return self._evaluate(state)
        except Exception:
            return self._evaluate(state)

        moves = state.get_valid_moves()
        if depth == 0 or not moves:
            return self._evaluate(state)

        # Max node
        if maximizing:
            value = -1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt is not None:
                    # Reverse perspective for opponent turn
                    if hasattr(nxt, "reverse_perspective"):
                        nxt.reverse_perspective()

                    score = self._alpha_beta(
                        nxt, depth - 1, alpha, beta,
                        False, time_left, sensors
                    )
                    value = max(value, score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        # Min node (opponent)
        else:
            value = 1e9
            for m in moves:
                d, t = self._unpack(m)
                nxt = state.forecast_move(d, t)
                if nxt is not None:
                    if hasattr(nxt, "reverse_perspective"):
                        nxt.reverse_perspective()

                    score = self._alpha_beta(
                        nxt, depth - 1, alpha, beta,
                        True, time_left, sensors
                    )
                    value = min(value, score)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # ----------------------------------------------------------------------
    # Move Selection (Immediate Utility + Minimax)
    # ----------------------------------------------------------------------
    def _choose_move(self, board_state: board.Board, sensors, time_left: Callable):

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # -----------------------------------------------------------
        # FIRST SCORE USING IMMEDIATE UTILITY
        # -----------------------------------------------------------
        scored = [(self._immediate_utility(m, board_state, sensors), m)
                  for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        # -----------------------------------------------------------
        # Prioritize safe egg-laying moves (always strong)
        # -----------------------------------------------------------
        egg_moves = [
            m for (sc, m) in scored
            if "egg" in self._enum(m[1]).lower() and sc > 0
        ]
        if egg_moves:
            return egg_moves[0]

        # -----------------------------------------------------------
        # Limit minimax to top-K most promising options
        # -----------------------------------------------------------
        top_k = min(5, len(scored))

        # -----------------------------------------------------------
        # Time-aware minimax depth
        # -----------------------------------------------------------
        try:
            tl = time_left()
        except Exception:
            tl = 999

        # Balanced depth:
        if tl > 120:
            depth = 3
        elif tl > 60:
            depth = 2
        else:
            depth = 1

        # -----------------------------------------------------------
        # Minimax evaluation
        # -----------------------------------------------------------
        best_move = scored[0][1]
        best_score = -1e9

        for i in range(top_k):
            imm, move = scored[i]
            d, t = self._unpack(move)

            nxt = board_state.forecast_move(d, t)

            if nxt is None:
                total = imm
            else:
                if hasattr(nxt, "reverse_perspective"):
                    nxt.reverse_perspective()

                fut = self._alpha_beta(
                    nxt,
                    depth - 1,
                    -1e9,
                    1e9,
                    False,      # opponent's turn
                    time_left,
                    sensors
                )

                total = imm + self.gamma * fut

            if total > best_score:
                best_score = total
                best_move = move

        return best_move
    # ----------------------------------------------------------------------
    # Main PLAY function — called every turn
    # ----------------------------------------------------------------------

    def play(self, board_state: board.Board, sensors, time_left: Callable):

        # Advance turn counter
        self.turn_index += 1

        # -----------------------------------------------------------
        # RECORD NEWLY DISCOVERED TRAPDOORS
        # -----------------------------------------------------------
        known = self._extract_known_traps(board_state, sensors)

        for loc in known:
            if loc not in self.confirmed_trapdoors:
                self._record_trapdoor_location(loc)

        # -----------------------------------------------------------
        # GET VALID MOVES
        # -----------------------------------------------------------
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # Current position
        loc = board_state.chicken_player.get_location()
        x0, y0 = loc

        # -----------------------------------------------------------
        # DETECT EGG PARITY ON FIRST EGG MOVE
        # -----------------------------------------------------------
        if self.my_egg_parity is None:
            for mv in moves:
                _, mt = self._unpack(mv)
                if "egg" in self._enum(mt).lower():
                    self.my_egg_parity = (x0 + y0) % 2
                    break

        # -----------------------------------------------------------
        # CHOOSE FINAL MOVE (utility + minimax)
        # -----------------------------------------------------------
        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            chosen = moves[0]

        direction, move_type = self._unpack(chosen)

        # Compute destination
        nx, ny = self._apply_dir(loc, direction)
        dest = (nx, ny)

        # -----------------------------------------------------------
        # MOVEMENT MEMORY UPDATE
        # -----------------------------------------------------------
        self.prev_loc = loc
        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1

        # Update recent list
        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        # -----------------------------------------------------------
        # RECORD EGG LAYING FOR LOOP AVOIDANCE
        # -----------------------------------------------------------
        if "egg" in self._enum(move_type).lower():
            self.egg_squares.add(dest)

        # -----------------------------------------------------------
        # EDGE-TRACKING UPDATE
        # -----------------------------------------------------------
        side = self._edge_side(nx, ny)
        if side is not None:
            self.edge_side_counts[side] += 1
            self.last_edge_side = side

        return chosen
