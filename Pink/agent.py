from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – PINK (Strategic Turds + Never Revisit)
# ================================================================
class PlayerAgent:
    """
    PINK:
    - Hard no-revisit rule for exploration
    - STRATEGIC turd placement (save for mid/late game)
    - Block opponent paths to corners
    - Block when opponent is nearby and ahead
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.my_egg_parity: Optional[int] = None

        # Trapdoor beliefs
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Safety confidence from sensors
        self.safety_confidence = np.zeros((self.board_size, self.board_size))

        # Confirmed traps
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()

        # VISITED TILES - will be EXCLUDED from move choices
        self.visited: Set[Tuple[int, int]] = set()

        # Direction tracking
        self.last_direction: Optional[Direction] = None

        self.turn_index = 0
        
        # Track egg counts for strategic decisions
        self.my_eggs = 0
        self.enemy_eggs = 0

    def _init_trapdoor_beliefs(self):
        center = self.board_size // 2
        for y in range(self.board_size):
            for x in range(self.board_size):
                dist = abs(x - center + 0.5) + abs(y - center + 0.5)
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 2.0 if dist <= 3 else 0.3
                else:
                    self.belief_black[y][x] = 2.0 if dist <= 3 else 0.3
        self._normalize()

    def _normalize(self):
        w_sum = self.belief_white.sum()
        b_sum = self.belief_black.sum()
        if w_sum > 1e-12:
            self.belief_white /= w_sum
        if b_sum > 1e-12:
            self.belief_black /= b_sum

    def _trapdoor_risk(self, x, y) -> float:
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return float(self.belief_white[y][x] + self.belief_black[y][x])
        return 1.0

    def _apply_dir(self, loc: Tuple[int, int], d: Direction) -> Tuple[int, int]:
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

    def _opposite_dir(self, d: Direction) -> Direction:
        if d == Direction.UP:
            return Direction.DOWN
        elif d == Direction.DOWN:
            return Direction.UP
        elif d == Direction.LEFT:
            return Direction.RIGHT
        elif d == Direction.RIGHT:
            return Direction.LEFT
        return d

    def _record_trap(self, loc):
        self.confirmed_trapdoors.add(loc)
        x, y = loc
        self.belief_white[y][x] = 0
        self.belief_black[y][x] = 0
        self.safety_confidence[y][x] = -999
        self._normalize()

    def _bayes_update(self, loc, sensors):
        (hw, fw), (hb, fb) = sensors
        lx, ly = loc

        self._update_belief_grid(self.belief_white, 0, loc, hw, fw)
        self._update_belief_grid(self.belief_black, 1, loc, hb, fb)
        self._normalize()

        if hw or fw or hb or fb:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        dist = abs(dx) + abs(dy)
                        if dist <= 2 and dist > 0:
                            penalty = -5.0 if (fw or fb) else -2.0
                            self.safety_confidence[ny][nx] += penalty / dist
        else:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        dist = abs(dx) + abs(dy)
                        if dist <= 2:
                            boost = 6.0 if dist <= 1 else 3.0
                            self.safety_confidence[ny][nx] += boost
                            factor = 0.05 if dist <= 1 else 0.3
                            self.belief_white[ny][nx] *= factor
                            self.belief_black[ny][nx] *= factor
            self._normalize()

        self.safety_confidence[ly][lx] = 100
        self.belief_white[ly][lx] = 0
        self.belief_black[ly][lx] = 0

    def _update_belief_grid(self, grid, parity, loc, heard, felt):
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
        if dx + dy == 1:
            return 0.50, 0.30
        if dx == 1 and dy == 1:
            return 0.25, 0.15
        if dx + dy == 2:
            return 0.10, 0.00
        return 0.00, 0.00

    # =================================================================
    # STRATEGIC TURD EVALUATION
    # =================================================================
    def _is_good_turd_timing(self, board_state) -> bool:
        """Should we consider placing a turd now?"""
        turds_left = board_state.chicken_player.get_turds_left()
        
        # No turds left
        if turds_left <= 0:
            return False
        
        # SAVE turds until turn 15+ (mid-game)
        if self.turn_index < 15:
            return False
        
        # Always consider in late game (turn 25+)
        if self.turn_index >= 25:
            return True
        
        # Mid-game: only turd if behind or opponent is close
        ox, oy = board_state.chicken_enemy.get_location()
        cx, cy = board_state.chicken_player.get_location()
        dist = abs(cx - ox) + abs(cy - oy)
        
        # Opponent nearby (can actually be blocked)
        if dist <= 5:
            return True
        
        # We're behind - need to slow them down
        if self.enemy_eggs > self.my_eggs + 2:
            return True
        
        return False

    def _score_turd_placement(self, cx, cy, board_state) -> float:
        """Score how good this turd placement is"""
        ox, oy = board_state.chicken_enemy.get_location()
        dist = abs(cx - ox) + abs(cy - oy)
        
        score = 0.0
        
        # === BLOCK OPPONENT'S EGG SQUARES ===
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (cx + cy) % 2 == opp_parity:
                score += 100  # Blocking their egg square
                
                # HUGE bonus for blocking opponent's corner
                if (cx in (0, self.board_size - 1)) and (cy in (0, self.board_size - 1)):
                    score += 250
        
        # === BLOCK OPPONENT'S PATH ===
        # Good if between opponent and a corner they might want
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner_x, corner_y in corners:
            # Is opponent closer to this corner than us?
            opp_to_corner = abs(ox - corner_x) + abs(oy - corner_y)
            us_to_corner = abs(cx - corner_x) + abs(cy - corner_y)
            
            # Are we between opponent and corner?
            turd_to_corner = abs(cx - corner_x) + abs(cy - corner_y)
            if turd_to_corner < opp_to_corner and turd_to_corner <= 3:
                score += 80  # Blocking path to corner
        
        # === PROXIMITY BONUS ===
        # Turds are more effective when opponent is nearby
        if dist <= 3:
            score += 120  # Very close - immediate impact
        elif dist <= 5:
            score += 60   # Close enough to matter
        elif dist > 7:
            score -= 50   # Too far - won't affect them
        
        # === SAME ROW/COLUMN BONUS ===
        # Creates a wall effect
        if cx == ox or cy == oy:
            score += 50
        
        # === EDGE PLACEMENT ===
        # Turds on edges create harder-to-bypass walls
        if cx == 0 or cx == self.board_size - 1 or cy == 0 or cy == self.board_size - 1:
            score += 30
        
        # === DEFENSIVE BONUS (when behind) ===
        if self.enemy_eggs > self.my_eggs:
            score += 40  # Extra incentive to slow them down
        
        return score

    def _score_move(self, move, board_state, cx, cy, is_revisit: bool) -> float:
        """Score a move - revisits get massive penalty"""
        d, mt = move
        nx, ny = self._apply_dir((cx, cy), d)

        # HARD BAN on confirmed traps
        if (nx, ny) in self.confirmed_trapdoors:
            return -1e12

        risk = self._trapdoor_risk(nx, ny)
        confidence = self.safety_confidence[ny][nx]

        # REVISIT PENALTY
        if is_revisit:
            return -1e6 + confidence

        util = 0.0

        # ===== EGG PRIORITY =====
        if mt == MoveType.EGG:
            util += 500
            if (cx in (0, self.board_size - 1)) and (cy in (0, self.board_size - 1)):
                util += 300

        # ===== SAFETY SCORING =====
        if risk > 0.12 and confidence < 5:
            util -= 400
        elif risk > 0.08 and confidence < 0:
            util -= 200

        util += min(confidence * 3, 100)
        util -= risk * 80

        # ===== DIRECTION MOMENTUM =====
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 50
            elif d == self._opposite_dir(self.last_direction):
                util -= 100

        # ===== STRATEGIC TURD SCORING =====
        if mt == MoveType.TURD:
            if not self._is_good_turd_timing(board_state):
                # NOT a good time for turds - heavy penalty
                util -= 500
            else:
                # Good timing - evaluate placement quality
                turd_score = self._score_turd_placement(cx, cy, board_state)
                util += turd_score
                
                # But still prefer eggs and exploration over turds
                util -= 50  # Small base penalty so we don't turd when we could egg

        return util + np.random.random() * 0.01

    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()

        # Separate moves into NEW tiles vs REVISITS
        new_moves = []
        revisit_moves = []

        for m in moves:
            d, mt = m
            nx, ny = self._apply_dir((cx, cy), d)
            if (nx, ny) in self.visited:
                revisit_moves.append(m)
            else:
                new_moves.append(m)

        # PRIORITY 1: EGG moves to NEW tiles
        egg_new = [m for m in new_moves if m[1] == MoveType.EGG]
        if egg_new:
            scored = [(self._score_move(m, board_state, cx, cy, False), m) for m in egg_new]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > -300:
                return scored[0][1]

        # PRIORITY 2: Any move to NEW tiles (including strategic turds)
        if new_moves:
            scored = [(self._score_move(m, board_state, cx, cy, False), m) for m in new_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > -300:
                return scored[0][1]

        # PRIORITY 3: EGG moves even if revisit
        egg_revisit = [m for m in revisit_moves if m[1] == MoveType.EGG]
        if egg_revisit:
            scored = [(self._score_move(m, board_state, cx, cy, True), m) for m in egg_revisit]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # PRIORITY 4: Forced to revisit - pick safest
        if revisit_moves:
            scored = [(self._score_move(m, board_state, cx, cy, True), m) for m in revisit_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        return moves[0]

    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        # Track egg counts
        self.my_eggs = board_state.chicken_player.get_eggs_laid()
        self.enemy_eggs = board_state.chicken_enemy.get_eggs_laid()

        # Record confirmed traps
        try:
            for loc in board_state.found_trapdoors:
                if loc not in self.confirmed_trapdoors:
                    self._record_trap(loc)
        except:
            pass

        cx, cy = board_state.chicken_player.get_location()
        self._bayes_update((cx, cy), sensors)
        self.visited.add((cx, cy))

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        if self.my_egg_parity is None:
            for m in moves:
                if m[1] == MoveType.EGG:
                    self.my_egg_parity = (cx + cy) % 2
                    break

        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            return moves[0]

        self.last_direction = chosen[0]
        return chosen
