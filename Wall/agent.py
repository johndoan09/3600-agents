from collections.abc import Callable
from typing import List, Tuple, Optional, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – WALL (Barrier-Based Turd Placement)
# ================================================================
class PlayerAgent:
    """
    WALL: Same as Void but with BARRIER-BASED turd placement.
    
    Creates WALLS that partition the board instead of
    individual blockers near the opponent.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.my_egg_parity: Optional[int] = None

        # PROPER BAYESIAN BELIEFS: probability that each tile has a trap
        self.trap_belief = np.zeros((self.board_size, self.board_size))
        self._init_prior_beliefs()
        
        # MEMORY: Track belief history for each tile
        self.belief_history = [[[] for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # Track number of observations for each tile
        self.observation_count = np.zeros((self.board_size, self.board_size))
        
        # Track cumulative evidence
        self.cumulative_evidence = np.zeros((self.board_size, self.board_size))
        
        # Confirmed traps and safe tiles
        self.confirmed_traps: Set[Tuple[int, int]] = set()
        self.confirmed_safe: Set[Tuple[int, int]] = set()

        # Visited tiles
        self.visited: Set[Tuple[int, int]] = set()
        
        # Opponent visited (confirmed safe)
        self.opponent_visited: Set[Tuple[int, int]] = set()

        # Direction tracking
        self.last_direction: Optional[Direction] = None

        self.turn_index = 0
        self.my_eggs = 0
        self.enemy_eggs = 0
        self.enemy_loc: Optional[Tuple[int, int]] = None
        
        # BLOCKING STRATEGY: Track egg positions
        self.my_egg_positions: Set[Tuple[int, int]] = set()
        self.enemy_egg_positions: Set[Tuple[int, int]] = set()
        
        # Movement tracking for trap detection
        self.last_position: Optional[Tuple[int, int]] = None
        self.intended_position: Optional[Tuple[int, int]] = None
        self.spawn_position: Optional[Tuple[int, int]] = None

    def _init_prior_beliefs(self):
        """Initialize with center-biased priors"""
        center = self.board_size / 2 - 0.5
        for y in range(self.board_size):
            for x in range(self.board_size):
                dist_to_center = abs(x - center) + abs(y - center)
                if dist_to_center <= 2:
                    self.trap_belief[y][x] = 0.15
                elif dist_to_center <= 4:
                    self.trap_belief[y][x] = 0.08
                else:
                    self.trap_belief[y][x] = 0.02

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

    def _is_edge(self, x, y) -> bool:
        return x == 0 or x == self.board_size - 1 or y == 0 or y == self.board_size - 1

    def _get_distance_type(self, dx, dy) -> str:
        adx, ady = abs(dx), abs(dy)
        if adx + ady == 1:
            return "adjacent"
        elif adx == 1 and ady == 1:
            return "diagonal"
        elif adx + ady == 2:
            return "two_away"
        return "far"

    def _get_signal_probability(self, dist_type: str) -> Tuple[float, float]:
        if dist_type == "adjacent":
            return 0.50, 0.30
        elif dist_type == "diagonal":
            return 0.25, 0.15
        elif dist_type == "two_away":
            return 0.10, 0.00
        return 0.00, 0.00

    def _get_silence_probability(self, dist_type: str) -> float:
        p_hear, p_feel = self._get_signal_probability(dist_type)
        return (1 - p_hear) * (1 - p_feel)

    def _record_trap(self, loc):
        x, y = loc
        self.confirmed_traps.add(loc)
        self.trap_belief[y][x] = 1.0

    def _bayesian_update(self, loc, sensors):
        (hw, fw), (hb, fb) = sensors
        lx, ly = loc
        
        self.trap_belief[ly][lx] = 0.0
        self.cumulative_evidence[ly][lx] = 100
        self.confirmed_safe.add((lx, ly))
        self.visited.add((lx, ly))
        
        heard = hw or hb
        felt = fw or fb
        
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = lx + dx, ly + dy
                if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                    continue
                
                if (nx, ny) in self.confirmed_safe or (nx, ny) in self.confirmed_traps:
                    continue
                if (nx, ny) in self.visited or (nx, ny) in self.opponent_visited:
                    continue
                
                dist_type = self._get_distance_type(dx, dy)
                if dist_type == "far":
                    continue
                
                prior = self.trap_belief[ny][nx]
                p_hear, p_feel = self._get_signal_probability(dist_type)
                
                if heard or felt:
                    if felt:
                        p_signal_given_trap = 1 - (1 - p_hear) * (1 - p_feel)
                        self.cumulative_evidence[ny][nx] -= 30 if dist_type == "adjacent" else 15
                    else:
                        p_signal_given_trap = p_hear
                        self.cumulative_evidence[ny][nx] -= 15 if dist_type == "adjacent" else 8
                    
                    p_signal_given_no_trap = 0.1
                    likelihood_ratio = p_signal_given_trap / max(p_signal_given_no_trap, 0.01)
                    posterior = (prior * likelihood_ratio) / (prior * likelihood_ratio + (1 - prior))
                    self.trap_belief[ny][nx] = min(0.95, posterior)
                else:
                    p_silence_given_trap = self._get_silence_probability(dist_type)
                    p_silence_given_no_trap = 1.0
                    
                    numerator = p_silence_given_trap * prior
                    denominator = p_silence_given_trap * prior + p_silence_given_no_trap * (1 - prior)
                    
                    if denominator > 0:
                        posterior = numerator / denominator
                        self.trap_belief[ny][nx] = posterior
                    
                    evidence_boost = 12 if dist_type == "adjacent" else 8 if dist_type == "diagonal" else 5
                    self.cumulative_evidence[ny][nx] += evidence_boost
                
                self.observation_count[ny][nx] += 1
                self.belief_history[ny][nx].append(self.trap_belief[ny][nx])
                if len(self.belief_history[ny][nx]) > 10:
                    self.belief_history[ny][nx].pop(0)

    def _get_trap_risk(self, x, y) -> float:
        if (x, y) in self.confirmed_traps:
            return 1.0
        if (x, y) in self.confirmed_safe or (x, y) in self.visited or (x, y) in self.opponent_visited:
            return 0.0
        return self.trap_belief[y][x]

    def _get_confidence(self, x, y) -> float:
        if (x, y) in self.confirmed_safe or (x, y) in self.visited or (x, y) in self.opponent_visited:
            return 1.0
        if (x, y) in self.confirmed_traps:
            return 1.0
        
        obs = self.observation_count[y][x]
        obs_confidence = min(1.0, obs / 4.0)
        
        evidence = self.cumulative_evidence[y][x]
        if evidence > 20:
            obs_confidence = min(1.0, obs_confidence + 0.3)
        elif evidence > 10:
            obs_confidence = min(1.0, obs_confidence + 0.15)
        
        history = self.belief_history[y][x]
        if len(history) >= 3:
            avg_belief = sum(history) / len(history)
            if avg_belief < 0.05:
                obs_confidence = min(1.0, obs_confidence + 0.2)
        
        return obs_confidence
    
    def _count_safe_neighbors(self, x, y) -> int:
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if (nx, ny) in self.confirmed_safe or (nx, ny) in self.visited or (nx, ny) in self.opponent_visited:
                        count += 1
        return count
    
    def _is_in_safe_zone(self, x, y) -> bool:
        safe_neighbors = self._count_safe_neighbors(x, y)
        return safe_neighbors >= 4

    def _is_safe_enough(self, x, y, threshold=0.05) -> bool:
        if (x, y) in self.confirmed_traps:
            return False
        if (x, y) in self.confirmed_safe or (x, y) in self.visited or (x, y) in self.opponent_visited:
            return True
        
        risk = self._get_trap_risk(x, y)
        confidence = self._get_confidence(x, y)
        return risk < threshold and confidence > 0.6

    def _score_turd_placement(self, cx, cy) -> float:
        """
        HYBRID turd placement - combines proximity urgency with barrier building.
        
        IMPROVED EARLY GAME: Match Void's aggression from turn 12.
        """
        score = -40  # Even lower base penalty for more turds
        
        # 1. TIMING: Match Void's turn 12 start
        if self.turn_index < 12:
            return -500  # Don't place very early (was 14)
        elif self.turn_index < 18:
            score -= 10   # Slight discourage early (reduced from -20)
        elif self.turn_index > 25:
            score += 70   # Encourage mid-late game
        
        # ===== PROXIMITY BONUSES (STRONGER for early game) =====
        # Place turds near opponent for immediate impact
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            dist = abs(cx - ox) + abs(cy - oy)
            
            # STRONGER proximity bonuses to match Void
            if dist == 1:
                score += 350  # Adjacent! MUST place (was 300)
            elif dist == 2:
                score += 220  # Close, high priority (was 180)
            elif dist <= 3:
                score += 130  # Nearby, good (was 100)
            elif dist <= 5:
                score += 50   # Medium range (was 40)
            else:
                score -= 60   # Too far
        
        # ===== BARRIER BONUSES (Wall's specialty) =====
        # 2. BARRIER CREATION: Turds that connect to edges or other obstacles
        connects_to_edge = (cx == 0 or cx == self.board_size - 1 or 
                           cy == 0 or cy == self.board_size - 1)
        
        # Count obstacles adjacent to this position (eggs, turds, edges)
        adjacent_obstacles = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= self.board_size or ny < 0 or ny >= self.board_size:
                    adjacent_obstacles += 1
                elif (nx, ny) in self.my_egg_positions:
                    adjacent_obstacles += 1
        
        # Bonus for turds that extend barriers (reduced thresholds)
        if adjacent_obstacles >= 2:
            score += 150  # Creates/extends a wall!
        elif adjacent_obstacles >= 1:
            score += 80   # Connects to something
        
        # Edge bonus (starts a wall)
        if connects_to_edge:
            score += 100
        
        # 3. LINE BONUS: Create straight barriers
        horizontal_line = 0
        vertical_line = 0
        
        for offset in [-2, -1, 1, 2]:
            hx, hy = cx + offset, cy
            if 0 <= hx < self.board_size:
                if (hx, hy) in self.my_egg_positions or hx == 0 or hx == self.board_size - 1:
                    horizontal_line += 1
            else:
                horizontal_line += 1
            
            vx, vy = cx, cy + offset
            if 0 <= vy < self.board_size:
                if (vx, vy) in self.my_egg_positions or vy == 0 or vy == self.board_size - 1:
                    vertical_line += 1
            else:
                vertical_line += 1
        
        if horizontal_line >= 2 or vertical_line >= 2:
            score += 120  # Part of a line!
        
        # 4. PARTITION BONUS: Cut off opponent from board sections
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            
            # Check if turd blocks access to unexplored areas
            partition_bonus = 0
            if cy < oy:  # Above opponent
                tiles_blocked = sum(1 for y in range(cy) 
                                   for x in range(self.board_size) 
                                   if (x, y) not in self.visited)
                if tiles_blocked > 8:  # Lowered from 10
                    partition_bonus = max(partition_bonus, 100)
            elif cy > oy:  # Below opponent
                tiles_blocked = sum(1 for y in range(cy + 1, self.board_size) 
                                   for x in range(self.board_size) 
                                   if (x, y) not in self.visited)
                if tiles_blocked > 8:
                    partition_bonus = max(partition_bonus, 100)
            
            if cx < ox:  # Left of opponent
                tiles_blocked = sum(1 for x in range(cx) 
                                   for y in range(self.board_size) 
                                   if (x, y) not in self.visited)
                if tiles_blocked > 8:
                    partition_bonus = max(partition_bonus, 100)
            elif cx > ox:  # Right of opponent
                tiles_blocked = sum(1 for x in range(cx + 1, self.board_size) 
                                   for y in range(self.board_size) 
                                   if (x, y) not in self.visited)
                if tiles_blocked > 8:
                    partition_bonus = max(partition_bonus, 100)
            
            score += partition_bonus
        
        # 5. OPPONENT'S EGG PARITY: Block their egg spots
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (cx + cy) % 2 == opp_parity:
                score += 120
        
        # 6. CORNER BLOCKING: Block paths to corners
        corners = [(0, 0), (0, self.board_size - 1), 
                   (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            for corner_x, corner_y in corners:
                dist_opp_to_corner = abs(ox - corner_x) + abs(oy - corner_y)
                dist_turd_to_corner = abs(cx - corner_x) + abs(cy - corner_y)
                
                if dist_turd_to_corner < dist_opp_to_corner and dist_turd_to_corner <= 3:
                    score += 100  # Blocking corner access
        
        # 7. DON'T BLOCK OURSELVES
        if (cx, cy) in self.visited and len(self.visited) < 18:
            score -= 40
        
        # 8. COMPETITIVE BONUS: More aggressive when behind
        egg_diff = self.my_eggs - self.enemy_eggs
        if egg_diff < -2:
            score += 40
        elif egg_diff < 0:
            score += 20
        
        return score

    def _score_blocking_placement(self, nx, ny) -> float:
        score = 0.0
        
        corners = [(0, 0), (0, self.board_size - 1), 
                   (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            for cx, cy in corners:
                dist_opp_to_corner = abs(ox - cx) + abs(oy - cy)
                dist_egg_to_corner = abs(nx - cx) + abs(ny - cy)
                
                if dist_egg_to_corner < dist_opp_to_corner and dist_egg_to_corner <= 3:
                    score += 100
        
        line_eggs = 0
        for offset in [-2, -1, 1, 2]:
            if (nx + offset, ny) in self.my_egg_positions:
                line_eggs += 1
            if (nx, ny + offset) in self.my_egg_positions:
                line_eggs += 1
        score += line_eggs * 50
        
        if nx == 0 or nx == self.board_size - 1 or ny == 0 or ny == self.board_size - 1:
            score += 60
        
        if self.enemy_loc:
            dist_to_enemy = abs(nx - self.enemy_loc[0]) + abs(ny - self.enemy_loc[1])
            if dist_to_enemy <= 3:
                score += 80
            elif dist_to_enemy <= 5:
                score += 40
        
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (nx + ny) % 2 == opp_parity:
                score += 70
        
        return score

    def _score_move(self, move, board_state, cx, cy) -> float:
        d, mt = move
        nx, ny = self._apply_dir((cx, cy), d)

        if (nx, ny) in self.confirmed_traps:
            return -1e12

        risk = self._get_trap_risk(nx, ny)
        confidence = self._get_confidence(nx, ny)
        is_visited = (nx, ny) in self.visited
        is_opponent_visited = (nx, ny) in self.opponent_visited
        is_confirmed_safe = (nx, ny) in self.confirmed_safe or is_visited or is_opponent_visited
        
        util = 0.0
        
        late_game = self.turn_index > 28
        very_late_game = self.turn_index > 34
        
        if very_late_game:
            RISK_VERY_SAFE = 0.08
            RISK_SAFE = 0.12
            RISK_ACCEPTABLE = 0.16
            RISK_CAUTION = 0.20
            CONFIDENCE_HIGH = 0.40
            CONFIDENCE_GOOD = 0.30
            CONFIDENCE_OK = 0.20
        elif late_game:
            RISK_VERY_SAFE = 0.06
            RISK_SAFE = 0.09
            RISK_ACCEPTABLE = 0.12
            RISK_CAUTION = 0.16
            CONFIDENCE_HIGH = 0.45
            CONFIDENCE_GOOD = 0.35
            CONFIDENCE_OK = 0.25
        else:
            RISK_VERY_SAFE = 0.045
            RISK_SAFE = 0.065
            RISK_ACCEPTABLE = 0.095
            RISK_CAUTION = 0.13
            CONFIDENCE_HIGH = 0.55
            CONFIDENCE_GOOD = 0.42
            CONFIDENCE_OK = 0.32
        
        in_safe_zone = self._is_in_safe_zone(nx, ny)
        safe_neighbors = self._count_safe_neighbors(nx, ny)
        evidence = self.cumulative_evidence[ny][nx]
        observations = self.observation_count[ny][nx]
        
        # IMPROVED: Earlier bonuses for exploration
        early_game = self.turn_index <= 15
        mid_game = 15 < self.turn_index <= 25
        
        late_bonus = 1.3 if very_late_game else (1.15 if late_game else 1.0)
        early_bonus = 1.15 if early_game else (1.08 if mid_game else 1.0)  # NEW: early/mid boost
        
        if is_confirmed_safe:
            if not is_visited:
                util += int(2000 * late_bonus * early_bonus)  # BOOSTED (was 1800)
            else:
                util += 20
                
        elif risk < RISK_VERY_SAFE and confidence >= CONFIDENCE_HIGH:
            if not is_visited:
                util += int(1600 * late_bonus * early_bonus)  # BOOSTED (was 1400)
            else:
                util += 15
                
        elif risk < RISK_SAFE and confidence >= CONFIDENCE_GOOD:
            if not is_visited:
                util += int(1200 * late_bonus * early_bonus)  # BOOSTED (was 1000)
            else:
                util += 10
                
        elif risk < RISK_ACCEPTABLE and confidence >= CONFIDENCE_OK and safe_neighbors >= 2:
            if not is_visited:
                util += int(900 * late_bonus)  # BOOSTED (was 700)
            else:
                util += 8
                
        elif risk < RISK_CAUTION and in_safe_zone and evidence > 5:
            if not is_visited:
                util += int(600 * late_bonus)  # BOOSTED (was 400)
            else:
                util += 5
                
        elif risk < RISK_CAUTION and safe_neighbors >= 3 and observations >= 2:
            if not is_visited:
                util += int(450 * late_bonus)  # BOOSTED (was 300)
            else:
                util += 5
                
        else:
            risk_penalty = -500 - (risk * 2000)
            if late_game:
                risk_penalty = risk_penalty * 0.7
            util += risk_penalty
        
        if not is_confirmed_safe:
            if evidence > 25:
                util += 200
            elif evidence > 15:
                util += 100
            elif evidence > 8:
                util += 50
            elif evidence < -10:
                util -= 200

        if mt == MoveType.EGG:
            util += 1000  # BOOSTED (was 800)
            
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 500  # BOOSTED corner eggs (was 400)
            
            # NEW: Edge eggs are valuable
            elif nx == 0 or nx == self.board_size - 1 or ny == 0 or ny == self.board_size - 1:
                util += 150  # Edge egg bonus
            
            blocking_score = self._score_blocking_placement(nx, ny)
            util += blocking_score
            
            # NEW: Egg priority when behind
            egg_diff = self.my_eggs - self.enemy_eggs
            if egg_diff < -2:
                util += 200  # Prioritize eggs when behind
            elif egg_diff < 0:
                util += 100

        # IMPROVED EARLY GAME: Be more aggressive about exploration
        if self.turn_index <= 15:  # Shortened (was 20)
            if self._is_edge(nx, ny) and not is_visited:
                util += 120  # Increased edge bonus (was 100)
            elif not self._is_edge(nx, ny) and risk > 0.06:  # Slightly relaxed (was 0.05)
                util -= 100  # Reduced penalty (was -150)
        elif self.turn_index <= 25:  # NEW: Mid-game exploration bonus
            if not is_visited and is_confirmed_safe:
                util += 80  # Encourage safe exploration mid-game

        if self.last_direction is not None:
            if d == self.last_direction:
                util += 25
            elif d == self._opposite_dir(self.last_direction):
                util -= 40

        if is_confirmed_safe and not is_visited:
            unknown_nearby = 0
            for ddy in range(-2, 3):
                for ddx in range(-2, 3):
                    tx, ty = nx + ddx, ny + ddy
                    if 0 <= tx < self.board_size and 0 <= ty < self.board_size:
                        if self.observation_count[ty][tx] < 3:
                            unknown_nearby += 1
            util += unknown_nearby * 15  # BOOSTED (was 10)
        
        # NEW: Bonus for reaching egg parity tiles
        if self.my_egg_parity is not None:
            if (nx + ny) % 2 == self.my_egg_parity and not is_visited:
                util += 80  # Prefer tiles where we can lay eggs

        if mt == MoveType.TURD:
            turd_score = self._score_turd_placement(cx, cy)
            util += turd_score

        return util + np.random.random() * 0.01

    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()

        safe_new = []
        very_safe = []
        safe_explore = []
        cautious_explore = []
        safe_revisit = []
        risky = []
        
        late_game = self.turn_index > 28
        very_late_game = self.turn_index > 34
        
        if very_late_game:
            thresh_very_safe = (0.08, 0.40)
            thresh_safe = (0.12, 0.30)
            thresh_acceptable = (0.16, 0.20)
            thresh_caution = 0.20
        elif late_game:
            thresh_very_safe = (0.06, 0.45)
            thresh_safe = (0.09, 0.35)
            thresh_acceptable = (0.12, 0.25)
            thresh_caution = 0.16
        else:
            thresh_very_safe = (0.045, 0.55)
            thresh_safe = (0.065, 0.42)
            thresh_acceptable = (0.095, 0.32)
            thresh_caution = 0.13
        
        for m in moves:
            d, mt = m
            nx, ny = self._apply_dir((cx, cy), d)
            
            if (nx, ny) in self.confirmed_traps:
                continue
            
            is_confirmed_safe = (nx, ny) in self.confirmed_safe or (nx, ny) in self.visited or (nx, ny) in self.opponent_visited
            
            if is_confirmed_safe:
                if (nx, ny) not in self.visited:
                    safe_new.append(m)
                else:
                    safe_revisit.append(m)
            else:
                risk = self._get_trap_risk(nx, ny)
                confidence = self._get_confidence(nx, ny)
                in_safe_zone = self._is_in_safe_zone(nx, ny)
                safe_neighbors = self._count_safe_neighbors(nx, ny)
                evidence = self.cumulative_evidence[ny][nx]
                observations = self.observation_count[ny][nx]
                
                if risk < thresh_very_safe[0] and confidence >= thresh_very_safe[1]:
                    very_safe.append(m)
                elif risk < thresh_safe[0] and confidence >= thresh_safe[1]:
                    safe_explore.append(m)
                elif risk < thresh_acceptable[0] and confidence >= thresh_acceptable[1] and safe_neighbors >= 2:
                    cautious_explore.append(m)
                elif risk < thresh_caution and in_safe_zone and evidence > 5:
                    cautious_explore.append(m)
                elif risk < thresh_caution and safe_neighbors >= 3 and observations >= 2:
                    cautious_explore.append(m)
                else:
                    risky.append(m)

        if safe_new:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_new]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        if very_safe:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in very_safe]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        if safe_explore:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_explore]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        if cautious_explore:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in cautious_explore]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        if safe_revisit:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_revisit]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        if risky:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in risky]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > -300:
                return scored[0][1]
            for m in safe_revisit:
                return m

        return moves[0]

    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        self.my_eggs = board_state.chicken_player.get_eggs_laid()
        self.enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        
        cx, cy = board_state.chicken_player.get_location()
        
        if self.spawn_position is None:
            self.spawn_position = (cx, cy)
        
        if self.intended_position is not None and self.last_position is not None:
            intended = self.intended_position
            current = (cx, cy)
            
            if current == self.spawn_position and intended != self.spawn_position:
                self._record_trap(intended)
        
        try:
            self.enemy_loc = board_state.chicken_enemy.get_location()
            if self.enemy_loc is not None:
                ex, ey = self.enemy_loc
                if (ex, ey) not in self.opponent_visited:
                    self.opponent_visited.add((ex, ey))
                    self.trap_belief[ey][ex] = 0.0
                    self.confirmed_safe.add((ex, ey))
        except:
            self.enemy_loc = None

        try:
            for loc in board_state.found_trapdoors:
                if tuple(loc) not in self.confirmed_traps:
                    self._record_trap(tuple(loc))
        except:
            pass
        
        self._bayesian_update((cx, cy), sensors)

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

        if chosen[1] == MoveType.EGG:
            self.my_egg_positions.add((cx, cy))
        
        prev_enemy_eggs = len(self.enemy_egg_positions)
        if self.enemy_eggs > prev_enemy_eggs and self.enemy_loc:
            self.enemy_egg_positions.add(self.enemy_loc)

        self.last_position = (cx, cy)
        self.last_direction = chosen[0]
        self.intended_position = self._apply_dir((cx, cy), chosen[0])
        
        return chosen

