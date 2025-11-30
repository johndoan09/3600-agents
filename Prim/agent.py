from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set, Iterable
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – PRIM (Improved Strategic Agent)
# ================================================================
class PlayerAgent:
    """
    PRIM:
    - Fixed minimax evaluation
    - Corner egg prioritization (3x bonus!)
    - Mobility-aware movement
    - Improved turd strategy (block opponent egg squares)
    - Better balanced thresholds
    - Edge walking preference
    - Proper Bayesian trapdoor inference
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

        # Turn tracking
        self.turn_index = 0
        self.gamma = 0.35

        # Adjusted safety thresholds (less conservative)
        self.THRESH_EARLY = 0.08
        self.THRESH_MID = 0.05
        self.THRESH_LATE = 0.03

    # --------------------------------------------------------------
    # Trapdoor belief initialization
    # --------------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                # White parity trap (even squares)
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 1
                else:
                    self.belief_black[y][x] = 1
        self._normalize()

    def _normalize(self):
        w_sum = self.belief_white.sum()
        b_sum = self.belief_black.sum()
        if w_sum > 1e-12:
            self.belief_white /= w_sum
        if b_sum > 1e-12:
            self.belief_black /= b_sum

    # --------------------------------------------------------------
    # Trapdoor Danger
    # --------------------------------------------------------------
    def _trapdoor_risk(self, x, y):
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return float(self.belief_white[y][x] + self.belief_black[y][x])
        return 1.0  # Out of bounds is max risk

    # --------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------
    def _apply_dir(self, loc: Tuple[int, int], d) -> Tuple[int, int]:
        x, y = loc
        if d == Direction.UP:
            return x, y - 1
        elif d == Direction.DOWN:
            return x, y + 1
        elif d == Direction.LEFT:
            return x - 1, y
        elif d == Direction.RIGHT:
            return x + 1, y
        return x, y

    def _phase(self):
        if self.turn_index <= 12:
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
        return known

    def _record_trap(self, loc):
        self.confirmed_trapdoors.add(loc)
        x, y = loc
        # Mark adjacent squares as danger zone
        for (nx, ny) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                self.confirmed_danger_zone.add((nx, ny))
        # Zero out belief for confirmed trap
        if (x + y) % 2 == 0:
            self.belief_white[y][x] = 0
        else:
            self.belief_black[y][x] = 0
        self._normalize()

    # --------------------------------------------------------------
    # BAYESIAN UPDATE ON SENSORS
    # --------------------------------------------------------------
    def _bayes_update(self, loc, sensors):
        (hw, fw), (hb, fb) = sensors

        # Update for white trapdoor
        self._update_belief_grid(self.belief_white, 0, loc, hw, fw)
        # Update for black trapdoor
        self._update_belief_grid(self.belief_black, 1, loc, hb, fb)
        
        self._normalize()

        # No signal means nearby tiles are safer
        if not (hw or fw or hb or fb):
            lx, ly = loc
            for y in range(self.board_size):
                for x in range(self.board_size):
                    dist = abs(x - lx) + abs(y - ly)
                    if dist <= 2:
                        decay = 0.3 if dist <= 1 else 0.6
                        self.belief_white[y][x] *= decay
                        self.belief_black[y][x] *= decay
            self._normalize()

    def _update_belief_grid(self, grid, parity, loc, heard, felt):
        lx, ly = loc
        like = np.ones_like(grid)

        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 != parity:
                    continue
                p_h, p_f = self._signal_prob(loc, (x, y))
                like[y][x] *= (p_h if heard else (1 - p_h))
                like[y][x] *= (p_f if felt else (1 - p_f))

        grid *= like

    def _signal_prob(self, here, trap):
        hx, hy = here
        tx, ty = trap
        dx, dy = abs(hx - tx), abs(hy - ty)

        # Direct adjacency (Manhattan distance 1)
        if dx + dy == 1:
            return 0.50, 0.30
        # Diagonal adjacency
        if dx == 1 and dy == 1:
            return 0.25, 0.15
        # Distance 2 (only hear possible)
        if dx + dy == 2 or (dx == 2 and dy == 0) or (dx == 0 and dy == 2):
            return 0.10, 0.00
        return 0.00, 0.00

    # --------------------------------------------------------------
    # Mobility Calculation
    # --------------------------------------------------------------
    def _count_safe_exits(self, x, y, board_state) -> int:
        """Count how many safe moves exist from position (x, y)"""
        exits = 0
        th = self._safety_threshold()
        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            nx, ny = self._apply_dir((x, y), d)
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if not board_state.is_cell_blocked((nx, ny)):
                    risk = self._trapdoor_risk(nx, ny)
                    if risk < th * 1.5:  # Slightly relaxed for counting
                        exits += 1
        return exits

    # --------------------------------------------------------------
    # State Evaluation (for minimax)
    # --------------------------------------------------------------
    def _evaluate_state(self, board_state, sensors) -> float:
        """Evaluate board state value"""
        try:
            cx, cy = board_state.chicken_player.get_location()
        except:
            return 0.0

        value = 0.0

        # Egg count difference
        my_eggs = board_state.chicken_player.get_eggs_laid()
        opp_eggs = board_state.chicken_enemy.get_eggs_laid()
        value += 100 * (my_eggs - opp_eggs)

        # Position quality
        risk = self._trapdoor_risk(cx, cy)
        value -= 200 * risk

        # Mobility
        mobility = self._count_safe_exits(cx, cy, board_state)
        value += 15 * mobility

        # Edge bonus
        on_edge = (cx in (0, 7) or cy in (0, 7))
        is_corner = (cx in (0, 7) and cy in (0, 7))
        if on_edge and not is_corner:
            value += 10

        # Parity alignment
        if self.my_egg_parity is not None:
            if (cx + cy) % 2 == self.my_egg_parity:
                value += 8

        return value

    # --------------------------------------------------------------
    # Immediate Utility
    # --------------------------------------------------------------
    def _immediate_utility(self, move, board_state, sensors) -> float:
        d, mt = move
        cx, cy = board_state.chicken_player.get_location()
        nx, ny = self._apply_dir((cx, cy), d)
        dest = (nx, ny)

        risk = self._trapdoor_risk(nx, ny)
        th = self._safety_threshold()

        # ABSOLUTE BAN on high-risk squares and confirmed danger zones
        if risk > th:
            return -1e12
        if dest in self.confirmed_trapdoors:
            return -1e12
        if dest in self.confirmed_danger_zone and risk > th * 0.5:
            return -5e11

        util = 0.0
        phase = self._phase()

        # ===== MOBILITY =====
        mobility = self._count_safe_exits(nx, ny, board_state)
        if mobility == 0:
            util -= 300  # Dead end is very bad
        elif mobility == 1:
            util -= 80   # Only one way out
        else:
            util += 10 * mobility

        # ===== EXPLORATION =====
        if dest not in self.visited_counts:
            if phase == "early":
                util += 50
            elif phase == "mid":
                util += 80
            else:
                util += 30
        else:
            util -= 30 * self.visited_counts[dest]

        # ===== ANTI-LOOP =====
        if dest == self.prev_loc:
            util -= 60
        util -= 20 * self.recent_positions.count(dest)

        # ===== EDGE PREFERENCE =====
        on_edge = (nx in (0, self.board_size - 1) or ny in (0, self.board_size - 1))
        is_corner = (nx in (0, self.board_size - 1) and ny in (0, self.board_size - 1))
        if on_edge and not is_corner:
            util += 12

        # ===== EGG SCORING =====
        if mt == MoveType.EGG:
            base_val = 150

            if risk < th * 0.3:
                util += base_val
            elif risk < th * 0.6:
                util += base_val * 0.7
            else:
                util += base_val * 0.4

            # CORNER EGGS ARE WORTH 3 EXTRA POINTS!
            if is_corner:
                if risk < th * 0.5:
                    util += 400  # Huge priority for safe corner eggs
                elif risk < th * 0.8:
                    util += 200

            # Parity alignment bonus
            if self.my_egg_parity is not None:
                if ((cx + cy) % 2) == self.my_egg_parity:
                    util += 15

            # Avoid re-egging same squares (can't, but good for pathing)
            if dest in self.egg_squares:
                util -= 20

        # ===== TURD SCORING =====
        if mt == MoveType.TURD:
            ox, oy = board_state.chicken_enemy.get_location()
            dist = abs(cx - ox) + abs(cy - oy)

            # Determine opponent's egg parity
            opp_parity = None
            if self.my_egg_parity is not None:
                opp_parity = 1 - self.my_egg_parity

            # Block opponent's potential egg squares
            if opp_parity is not None and (cx + cy) % 2 == opp_parity:
                util += 70
                # Blocking opponent's corner is HUGE
                if cx in (0, self.board_size - 1) and cy in (0, self.board_size - 1):
                    util += 150

            # Tactical distance scoring
            if dist <= 1:
                util -= 100  # Can't place turd too close
            elif 2 <= dist <= 4:
                util += 40
                # Same row or column creates corridor block
                if cx == ox or cy == oy:
                    util += 30
            elif dist > 6:
                util -= 10  # Too far to matter much

            # Don't waste turds early
            if phase == "early" and self.turn_index < 8:
                util -= 30

        # ===== CONTINUOUS RISK PENALTY =====
        util -= 120 * risk

        # ===== DISTANCE FROM CENTER (trapdoors are near center) =====
        center = self.board_size / 2 - 0.5
        dist_from_center = abs(nx - center) + abs(ny - center)
        if dist_from_center <= 2:
            util -= 15  # Slight penalty for center area

        return util + np.random.random() * 0.001

    # --------------------------------------------------------------
    # MINIMAX WITH PROPER EVALUATION
    # --------------------------------------------------------------
    def _minimax(self, state, move, sensors, depth=1) -> float:
        d, mt = move
        nxt = state.forecast_move(d, mt)
        if nxt is None:
            return -1e9

        # Base value from immediate utility
        base_val = self._immediate_utility(move, state, sensors)

        if depth <= 0:
            return base_val

        # Get opponent's possible moves
        opp_moves = nxt.get_valid_moves(enemy=True)
        if not opp_moves:
            # Opponent blocked - this is good!
            return base_val + 200

        # Minimax: assume opponent plays best response
        worst_future = float('inf')
        for opp_move in opp_moves[:6]:  # Limit branching
            od, omt = opp_move
            # Forecast opponent's move
            try:
                after_opp = nxt.forecast_move(od, omt, check_ok=False)
                if after_opp is None:
                    continue
                # Evaluate resulting state
                future_val = self._evaluate_state(after_opp, sensors)
                worst_future = min(worst_future, future_val)
            except:
                continue

        if worst_future == float('inf'):
            return base_val

        return base_val + self.gamma * worst_future

    # --------------------------------------------------------------
    # MOVE SELECTION
    # --------------------------------------------------------------
    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()
        th = self._safety_threshold()

        # HARD SAFETY FILTER — remove dangerous moves
        safe_moves = []
        for m in moves:
            d, mt = m
            nx, ny = self._apply_dir((cx, cy), d)
            r = self._trapdoor_risk(nx, ny)
            if r <= th and (nx, ny) not in self.confirmed_trapdoors:
                safe_moves.append(m)

        if safe_moves:
            moves = safe_moves
        else:
            # Extreme fallback: choose lowest-risk move
            moves = sorted(moves, key=lambda m: self._trapdoor_risk(
                *self._apply_dir((cx, cy), m[0])))

        # Prioritize EGG moves with quick evaluation
        egg_moves = [m for m in moves if m[1] == MoveType.EGG]
        if egg_moves:
            # Evaluate egg moves more carefully
            scored = [(self._immediate_utility(m, board_state, sensors), m)
                      for m in egg_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_egg = scored[0]
            if best_egg[0] > 50:  # Good egg move found
                return best_egg[1]

        # Check remaining time
        try:
            remaining = time_left()
            use_minimax = remaining > 5
        except:
            use_minimax = True

        # Score all moves
        if use_minimax and len(moves) <= 12:
            scored = [(self._minimax(board_state, m, sensors, depth=1), m)
                      for m in moves]
        else:
            scored = [(self._immediate_utility(m, board_state, sensors), m)
                      for m in moves]

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # --------------------------------------------------------------
    # MAIN API
    # --------------------------------------------------------------
    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        # Update confirmed traps
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

        # Detect egg parity on first opportunity
        if self.my_egg_parity is None:
            for m in moves:
                if m[1] == MoveType.EGG:
                    self.my_egg_parity = (cx + cy) % 2
                    break

        # Choose best move
        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            return moves[0]

        # Record movement
        d, mt = chosen
        nx, ny = self._apply_dir((cx, cy), d)

        self.prev_loc = (cx, cy)
        dest = (nx, ny)

        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1
        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        if mt == MoveType.EGG:
            self.egg_squares.add((cx, cy))  # Egg is laid at current position, not destination

        return chosen

