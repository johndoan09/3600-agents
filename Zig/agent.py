from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – ZIG (Alpha v2 with Blocking Strategy)
# ================================================================
class PlayerAgent:
    """
    ZIG: Alpha v2 with BLOCKING STRATEGY
    - All of Alpha's trapdoor detection and safety systems
    - NEW: Strategic egg placement that blocks opponent's parity
    - NEW: Prioritizes egg barriers and lines
    - NEW: Blocks opponent paths to corners
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
        
        # DANGER SIGNAL ACCUMULATOR
        self.danger_count = np.zeros((self.board_size, self.board_size))
        self.signal_processed_from: Set[Tuple[int, int]] = set()

        # Confirmed traps
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()

        # VISITED TILES
        self.visited: Set[Tuple[int, int]] = set()
        self.visit_time: Dict[Tuple[int, int], int] = {}

        # Direction tracking
        self.last_direction: Optional[Direction] = None
        
        # Loop detection
        self.recent_positions: List[Tuple[int, int]] = []
        self.turns_since_new_tile = 0
        self.desperation_level = 0

        self.turn_index = 0
        
        # Track egg counts
        self.my_eggs = 0
        self.enemy_eggs = 0
        
        # Track opponent
        self.enemy_loc: Optional[Tuple[int, int]] = None
        self.enemy_visited: Set[Tuple[int, int]] = set()
        self.last_enemy_loc: Optional[Tuple[int, int]] = None
        
        # NEW: Track our egg positions for blocking strategy
        self.my_egg_positions: Set[Tuple[int, int]] = set()

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
        self.danger_count[y][x] = 100
        self._normalize()

    def _bayes_update(self, loc, sensors):
        (hw, fw), (hb, fb) = sensors
        lx, ly = loc
        
        first_time_here = loc not in self.signal_processed_from
        self.signal_processed_from.add(loc)

        self._update_belief_grid(self.belief_white, 0, loc, hw, fw)
        self._update_belief_grid(self.belief_black, 1, loc, hb, fb)
        self._normalize()

        if hw or fw or hb or fb:
            felt_signal = fw or fb
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if (nx, ny) in self.visited:
                            continue
                        
                        dist = abs(dx) + abs(dy)
                        if dist <= 2 and dist > 0:
                            if felt_signal and dist == 1:
                                self.safety_confidence[ny][nx] -= 80
                                if first_time_here:
                                    self.danger_count[ny][nx] += 5
                            elif felt_signal and dist == 2:
                                self.safety_confidence[ny][nx] -= 25
                                if first_time_here:
                                    self.danger_count[ny][nx] += 2
                            elif dist == 1:
                                self.safety_confidence[ny][nx] -= 40
                                if first_time_here:
                                    self.danger_count[ny][nx] += 3
                            elif dist == 2:
                                self.safety_confidence[ny][nx] -= 15
                                if first_time_here:
                                    self.danger_count[ny][nx] += 2
        else:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        dist = abs(dx) + abs(dy)
                        if dist <= 2:
                            boost = 15.0 if dist <= 1 else 8.0
                            self.safety_confidence[ny][nx] += boost
                            self.danger_count[ny][nx] = max(0, self.danger_count[ny][nx] - 1)
                            factor = 0.02 if dist <= 1 else 0.2
                            self.belief_white[ny][nx] *= factor
                            self.belief_black[ny][nx] *= factor
            self._normalize()

        self.safety_confidence[ly][lx] = 100
        self.danger_count[ly][lx] = 0
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
    # LOOP DETECTION (same as Alpha)
    # =================================================================
    def _is_stuck_in_loop(self) -> bool:
        if self.turns_since_new_tile >= 4:
            return True
        if len(self.recent_positions) < 8:
            return False
        recent = self.recent_positions[-8:]
        unique = set(recent)
        if len(unique) <= 4:
            return True
        from collections import Counter
        counts = Counter(recent)
        if any(c >= 3 for c in counts.values()):
            return True
        return False
    
    def _is_oscillating(self) -> bool:
        if len(self.recent_positions) < 4:
            return False
        last4 = self.recent_positions[-4:]
        if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
            return True
        if len(self.recent_positions) >= 3:
            last3 = self.recent_positions[-3:]
            if last3[0] == last3[2] and last3[0] != last3[1]:
                return True
        if len(self.recent_positions) >= 6:
            last6 = self.recent_positions[-6:]
            if len(set(last6)) <= 3:
                return True
        if len(self.recent_positions) >= 8:
            last8 = self.recent_positions[-8:]
            if len(set(last8)) <= 4:
                return True
        return False
    
    def _get_oscillation_tiles(self) -> Set[Tuple[int, int]]:
        if len(self.recent_positions) < 4:
            return set()
        last6 = self.recent_positions[-6:] if len(self.recent_positions) >= 6 else self.recent_positions[-4:]
        return set(last6)
    
    def _distance_to_center(self, x, y) -> float:
        center = self.board_size / 2 - 0.5
        return abs(x - center) + abs(y - center)
    
    def _count_unvisited_nearby(self, x, y) -> int:
        count = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if (nx, ny) not in self.visited:
                        count += 1
        return count

    # =================================================================
    # NEW: BLOCKING STRATEGY - Strategic Egg Placement
    # =================================================================
    def _score_egg_blocking(self, nx: int, ny: int) -> float:
        """Score how well this egg placement blocks the opponent"""
        if self.my_egg_parity is None:
            return 0.0
        
        score = 0.0
        opp_parity = 1 - self.my_egg_parity
        
        # BONUS 1: Egg on opponent's parity = blocks their egg square
        # (This happens when we move to opponent parity tile and CAN'T lay egg,
        # but the NEXT tile where we CAN lay egg blocks a diagonal)
        # Actually, we can only lay eggs on OUR parity, so blocking is about position
        
        # BONUS 2: Egg creates a LINE with existing eggs (barrier)
        line_bonus = self._count_eggs_in_line(nx, ny)
        score += line_bonus * 40  # Each egg in line adds 40
        
        # BONUS 3: Egg on edge = creates perimeter barrier
        if nx == 0 or nx == self.board_size - 1:
            score += 50
        if ny == 0 or ny == self.board_size - 1:
            score += 50
        
        # BONUS 4: Egg blocks path to corner
        corner_block = self._blocks_corner_path(nx, ny)
        score += corner_block
        
        # BONUS 5: Egg is near opponent (more disruptive)
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            dist = abs(nx - ox) + abs(ny - oy)
            if dist <= 3:
                score += 60  # Close to opponent = more blocking value
            elif dist <= 5:
                score += 30
        
        # BONUS 6: Egg extends our territory
        territory_score = self._territory_expansion(nx, ny)
        score += territory_score
        
        return score
    
    def _count_eggs_in_line(self, x: int, y: int) -> int:
        """Count how many of our eggs are in a line (row or column) with this position"""
        row_count = 0
        col_count = 0
        
        for ex, ey in self.my_egg_positions:
            if ey == y:  # Same row
                row_count += 1
            if ex == x:  # Same column
                col_count += 1
        
        return max(row_count, col_count)
    
    def _blocks_corner_path(self, x: int, y: int) -> float:
        """Check if this egg blocks opponent's path to a corner"""
        if self.enemy_loc is None:
            return 0.0
        
        ox, oy = self.enemy_loc
        score = 0.0
        
        corners = [(0, 0), (0, self.board_size - 1), 
                   (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        
        for cx, cy in corners:
            # Is opponent closer to this corner than us?
            opp_dist = abs(ox - cx) + abs(oy - cy)
            egg_dist = abs(x - cx) + abs(y - cy)
            
            # If egg is between opponent and corner
            if egg_dist < opp_dist:
                # Check if egg is roughly on the path
                if (min(ox, cx) <= x <= max(ox, cx) and 
                    min(oy, cy) <= y <= max(oy, cy)):
                    score += 80  # Strong blocking bonus
        
        return score
    
    def _territory_expansion(self, x: int, y: int) -> float:
        """Score based on how much this egg expands our controlled territory"""
        if len(self.my_egg_positions) == 0:
            return 0.0
        
        # Calculate bounding box of our eggs
        min_x = min(ex for ex, ey in self.my_egg_positions)
        max_x = max(ex for ex, ey in self.my_egg_positions)
        min_y = min(ey for ex, ey in self.my_egg_positions)
        max_y = max(ey for ex, ey in self.my_egg_positions)
        
        score = 0.0
        
        # Bonus for expanding the bounding box
        if x < min_x or x > max_x:
            score += 30  # Expands horizontally
        if y < min_y or y > max_y:
            score += 30  # Expands vertically
        
        # Bonus for connecting to existing eggs (adjacent)
        for ex, ey in self.my_egg_positions:
            if abs(x - ex) + abs(y - ey) == 2:  # Diagonal neighbor (same parity)
                score += 20
        
        return score

    # =================================================================
    # STRATEGIC TURD EVALUATION (same as Alpha)
    # =================================================================
    def _is_good_turd_timing(self, board_state) -> bool:
        turds_left = board_state.chicken_player.get_turds_left()
        if turds_left <= 0:
            return False
        if self.turn_index < 15:
            return False
        if self.turn_index >= 25:
            return True
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            cx, cy = board_state.chicken_player.get_location()
            dist = abs(cx - ox) + abs(cy - oy)
            if dist <= 5:
                return True
        if self.enemy_eggs > self.my_eggs + 2:
            return True
        return False

    def _score_turd_placement(self, cx, cy, board_state) -> float:
        if self.enemy_loc is None:
            return 0.0
        ox, oy = self.enemy_loc
        dist = abs(cx - ox) + abs(cy - oy)
        
        score = 0.0
        
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (cx + cy) % 2 == opp_parity:
                score += 100
                if (cx in (0, self.board_size - 1)) and (cy in (0, self.board_size - 1)):
                    score += 250
        
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner_x, corner_y in corners:
            opp_to_corner = abs(ox - corner_x) + abs(oy - corner_y)
            turd_to_corner = abs(cx - corner_x) + abs(cy - corner_y)
            if turd_to_corner < opp_to_corner and turd_to_corner <= 3:
                score += 80
        
        if dist <= 3:
            score += 120
        elif dist <= 5:
            score += 60
        elif dist > 7:
            score -= 50
        
        if cx == ox or cy == oy:
            score += 50
        
        if cx == 0 or cx == self.board_size - 1 or cy == 0 or cy == self.board_size - 1:
            score += 30
        
        if self.enemy_eggs > self.my_eggs:
            score += 40
        
        return score

    def _score_move(self, move, board_state, cx, cy, is_revisit: bool) -> float:
        """Score a move with blocking strategy"""
        d, mt = move
        nx, ny = self._apply_dir((cx, cy), d)

        # HARD BAN on confirmed traps
        if (nx, ny) in self.confirmed_trapdoors:
            return -1e12

        risk = self._trapdoor_risk(nx, ny)
        confidence = self.safety_confidence[ny][nx]
        danger = self.danger_count[ny][nx]

        # REVISIT PENALTY
        if is_revisit:
            base_penalty = -1e6
            
            if self._is_oscillating():
                oscillation_tiles = self._get_oscillation_tiles()
                if (nx, ny) in oscillation_tiles:
                    return -1e9
            
            if self._is_stuck_in_loop():
                unvisited_nearby = self._count_unvisited_nearby(nx, ny)
                escape_bonus = unvisited_nearby * 60
                last_visit = self.visit_time.get((nx, ny), 0)
                recency_bonus = (self.turn_index - last_visit) * 8
                center_bonus = 0
                current_center_dist = self._distance_to_center(cx, cy)
                new_center_dist = self._distance_to_center(nx, ny)
                if new_center_dist < current_center_dist and risk < 0.06 and danger < 2:
                    center_bonus = 20
                safety_penalty = risk * 150 + danger * 50
                return base_penalty + confidence + escape_bonus + recency_bonus + center_bonus - safety_penalty
            return base_penalty + confidence

        util = 0.0

        # ===== EGG PRIORITY WITH BLOCKING STRATEGY =====
        if mt == MoveType.EGG:
            util += 500
            # Corner eggs are very valuable
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 300
            
            # NEW: Add blocking strategy bonus
            blocking_score = self._score_egg_blocking(nx, ny)
            util += blocking_score

        # ===== SAFETY SCORING (same as Alpha) =====
        tiles_explored = len(self.visited)
        knowledge_factor = min(1.0, tiles_explored / 25.0)
        
        egg_deficit = self.enemy_eggs - self.my_eggs
        competitive_bonus = 0.0
        if egg_deficit >= 3:
            competitive_bonus = min(0.3, egg_deficit * 0.05)
        
        confidence_from_knowledge = knowledge_factor * 0.3
        desperation_bonus = self.desperation_level * 0.07
        
        if confidence > 20:
            desperation_factor = max(0.3, 1.0 - confidence_from_knowledge - desperation_bonus - competitive_bonus)
        else:
            desperation_factor = max(0.4, 1.0 - desperation_bonus * 0.5 - competitive_bonus * 0.5)
        
        if danger >= 5:
            util -= int(1200 * desperation_factor)
        elif danger >= 3:
            util -= int(800 * desperation_factor)
        elif danger >= 2:
            util -= int(500 * desperation_factor)
        elif danger >= 1:
            util -= int(200 * desperation_factor)
        
        if confidence < -50:
            util -= int(600 * desperation_factor)
        elif confidence < -20:
            util -= int(400 * desperation_factor)
        elif confidence < 0:
            util -= int(200 * desperation_factor)
        
        if risk > 0.12 and confidence < 5:
            util -= int(600 * desperation_factor)
        elif risk > 0.08 and confidence < 0:
            util -= int(400 * desperation_factor)
        elif risk > 0.10:
            util -= int(200 * desperation_factor)
        
        if confidence > 0:
            util += min(confidence * 2, 60)
        util -= risk * 120
        
        if (nx, ny) in self.enemy_visited:
            util += 150

        # ===== DIRECTION MOMENTUM =====
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 35
            elif d == self._opposite_dir(self.last_direction):
                util -= 70

        # ===== LOOP ESCAPE =====
        if self._is_stuck_in_loop():
            unvisited_nearby = self._count_unvisited_nearby(nx, ny)
            util += unvisited_nearby * 50
            
            current_center_dist = self._distance_to_center(cx, cy)
            new_center_dist = self._distance_to_center(nx, ny)
            if new_center_dist < current_center_dist:
                if risk < 0.05 and confidence > 0 and danger < 2:
                    util += 100
                elif risk < 0.08 and danger < 2:
                    util += 30
            
            if nx == 0 or nx == self.board_size - 1 or ny == 0 or ny == self.board_size - 1:
                util -= 40

        # ===== STRATEGIC TURD SCORING =====
        if mt == MoveType.TURD:
            if not self._is_good_turd_timing(board_state):
                util -= 500
            else:
                turd_score = self._score_turd_placement(cx, cy, board_state)
                util += turd_score
                util -= 50

        return util + np.random.random() * 0.01

    def _pick_with_randomness(self, scored_moves, randomness_chance=0.15):
        if len(scored_moves) >= 2 and np.random.random() < randomness_chance:
            if scored_moves[0][0] - scored_moves[1][0] < 100:
                return scored_moves[1][1]
        return scored_moves[0][1]

    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()

        new_moves = []
        revisit_moves = []

        for m in moves:
            d, mt = m
            nx, ny = self._apply_dir((cx, cy), d)
            if (nx, ny) in self.visited:
                revisit_moves.append(m)
            else:
                new_moves.append(m)

        acceptance_threshold = -300
        
        egg_deficit = self.enemy_eggs - self.my_eggs
        if egg_deficit >= 5:
            acceptance_threshold = -600
        elif egg_deficit >= 3:
            acceptance_threshold = -450
        
        if self._is_oscillating():
            acceptance_threshold = -1000
        elif self._is_stuck_in_loop():
            acceptance_threshold = -700

        # PRIORITY 1: EGG moves to NEW tiles
        egg_new = [m for m in new_moves if m[1] == MoveType.EGG]
        if egg_new:
            scored = [(self._score_move(m, board_state, cx, cy, False), m) for m in egg_new]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > acceptance_threshold:
                return scored[0][1]

        # PRIORITY 2: Any move to NEW tiles
        if new_moves:
            scored = [(self._score_move(m, board_state, cx, cy, False), m) for m in new_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > acceptance_threshold:
                return self._pick_with_randomness(scored)

        # PRIORITY 3: EGG moves even if revisit
        egg_revisit = [m for m in revisit_moves if m[1] == MoveType.EGG]
        if egg_revisit:
            scored = [(self._score_move(m, board_state, cx, cy, True), m) for m in egg_revisit]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # PRIORITY 4: Forced to revisit
        if revisit_moves:
            scored = [(self._score_move(m, board_state, cx, cy, True), m) for m in revisit_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        return moves[0]

    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        # Track egg count changes to detect when we laid an egg
        old_eggs = self.my_eggs
        self.my_eggs = board_state.chicken_player.get_eggs_laid()
        self.enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        
        # Track our egg positions
        cx, cy = board_state.chicken_player.get_location()
        if self.my_eggs > old_eggs:
            # We just laid an egg at our previous position
            if len(self.recent_positions) > 0:
                last_pos = self.recent_positions[-1]
                self.my_egg_positions.add(last_pos)
        
        try:
            self.enemy_loc = board_state.chicken_enemy.get_location()
            
            if self.enemy_loc is not None:
                ex, ey = self.enemy_loc
                if (ex, ey) not in self.enemy_visited:
                    self.enemy_visited.add((ex, ey))
                    self.safety_confidence[ey][ex] = max(self.safety_confidence[ey][ex], 80)
                    self.danger_count[ey][ex] = 0
                    self.belief_white[ey][ex] = 0
                    self.belief_black[ey][ex] = 0
                    
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = ex + dx, ey + dy
                        if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                            self.safety_confidence[ny][nx] += 5
                            self.danger_count[ny][nx] = max(0, self.danger_count[ny][nx] - 1)
                
                self.last_enemy_loc = (ex, ey)
        except:
            self.enemy_loc = None

        try:
            for loc in board_state.found_trapdoors:
                if loc not in self.confirmed_trapdoors:
                    self._record_trap(loc)
        except:
            pass

        self._bayes_update((cx, cy), sensors)
        
        if (cx, cy) not in self.visited:
            self.turns_since_new_tile = 0
            self.desperation_level = max(0, self.desperation_level - 2)
        else:
            self.turns_since_new_tile += 1
            if self.turns_since_new_tile >= 3:
                self.desperation_level = min(10, self.desperation_level + 1)
            if self._is_oscillating():
                self.desperation_level = min(10, self.desperation_level + 3)
        
        self.visited.add((cx, cy))
        self.visit_time[(cx, cy)] = self.turn_index
        
        self.recent_positions.append((cx, cy))
        if len(self.recent_positions) > 12:
            self.recent_positions.pop(0)

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
