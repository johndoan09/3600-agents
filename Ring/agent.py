from collections.abc import Callable
from typing import List, Tuple, Optional, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – RING (Inside-Out Strategy)
# ================================================================
class PlayerAgent:
    """
    RING: Control inner rings first, block with turds, claim edges late.
    
    Strategy:
    1. EARLY GAME (turns 1-20): Explore Layer 1-2 (second/third outer rings)
    2. MID GAME (turns 15-30): Place turds in Layers 1-2 to block opponent
    3. LATE GAME (turns 25+): Claim Layer 0 (edges) - opponent can't reach them!
    
    Layer definitions (8x8 board):
    - Layer 0: Edge tiles (x=0,7 or y=0,7)
    - Layer 1: x=1,6 or y=1,6 (but not Layer 0)
    - Layer 2: x=2,5 or y=2,5 (but not Layers 0-1)
    - Layer 3+: Center tiles
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.my_egg_parity: Optional[int] = None

        # BAYESIAN BELIEFS
        self.trap_belief = np.zeros((self.board_size, self.board_size))
        self._init_prior_beliefs()
        
        self.belief_history = [[[] for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.observation_count = np.zeros((self.board_size, self.board_size))
        self.cumulative_evidence = np.zeros((self.board_size, self.board_size))
        
        self.confirmed_traps: Set[Tuple[int, int]] = set()
        self.confirmed_safe: Set[Tuple[int, int]] = set()
        self.visited: Set[Tuple[int, int]] = set()
        self.opponent_visited: Set[Tuple[int, int]] = set()

        self.last_direction: Optional[Direction] = None
        self.turn_index = 0
        self.my_eggs = 0
        self.enemy_eggs = 0
        self.enemy_loc: Optional[Tuple[int, int]] = None
        
        self.my_egg_positions: Set[Tuple[int, int]] = set()
        self.enemy_egg_positions: Set[Tuple[int, int]] = set()
        
        self.last_position: Optional[Tuple[int, int]] = None
        self.intended_position: Optional[Tuple[int, int]] = None
        self.spawn_position: Optional[Tuple[int, int]] = None
        
        # Board control tracking
        self.total_tiles = self.board_size * self.board_size
        self.board_control = 0.0
        
        # Turd tracking
        self.my_turd_positions: Set[Tuple[int, int]] = set()

    def _init_prior_beliefs(self):
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

    def _get_layer(self, x, y) -> int:
        """
        Get which layer a tile is in (0 = edge, 1 = second outer, etc.)
        """
        # Distance from each edge
        dist_left = x
        dist_right = self.board_size - 1 - x
        dist_top = y
        dist_bottom = self.board_size - 1 - y
        
        # Layer = minimum distance to any edge
        return min(dist_left, dist_right, dist_top, dist_bottom)

    def _is_layer_0(self, x, y) -> bool:
        """Layer 0 = Edge tiles"""
        return self._get_layer(x, y) == 0

    def _is_layer_1(self, x, y) -> bool:
        """Layer 1 = Second outer ring"""
        return self._get_layer(x, y) == 1

    def _is_layer_2(self, x, y) -> bool:
        """Layer 2 = Third outer ring"""
        return self._get_layer(x, y) == 2

    def _is_inner_ring(self, x, y) -> bool:
        """Layers 1-2 = Our target early game area"""
        layer = self._get_layer(x, y)
        return layer == 1 or layer == 2

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
        
        self.board_control = len(self.visited) / self.total_tiles
        
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
                    
                    evidence_boost = 14 if dist_type == "adjacent" else 9 if dist_type == "diagonal" else 6
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
            obs_confidence = min(1.0, obs_confidence + 0.30)
        elif evidence > 12:
            obs_confidence = min(1.0, obs_confidence + 0.18)
        elif evidence > 6:
            obs_confidence = min(1.0, obs_confidence + 0.10)
        
        history = self.belief_history[y][x]
        if len(history) >= 3:
            avg_belief = sum(history) / len(history)
            if avg_belief < 0.05:
                obs_confidence = min(1.0, obs_confidence + 0.20)
        
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

    def _lane_rank(self, x, y) -> float:
        """
        Returns how valuable this tile is for interior lanes (Layer 2 best).
        Used to mimic the black chicken's habit of patrolling the inner rings.
        """
        layer = self._get_layer(x, y)
        if layer == 2:
            return 1.0
        if layer == 3:
            return 0.6
        if layer == 1:
            return 0.35
        return 0.0

    def _forms_lane_barrier(self, x, y) -> bool:
        """
        Checks if dropping a turd here would complete a small barrier segment
        across a lane (row/column). Encourages inner-lane walls like the black chicken.
        """
        if self._lane_rank(x, y) < 0.9:
            return False
        
        lane_row_links = 0
        lane_col_links = 0
        
        for offset in (-2, -1, 1, 2):
            if (x + offset, y) in self.my_turd_positions and self._lane_rank(x + offset, y) >= 0.9:
                lane_row_links += 1
            if (x, y + offset) in self.my_turd_positions and self._lane_rank(x, y + offset) >= 0.9:
                lane_col_links += 1
        
        return lane_row_links >= 2 or lane_col_links >= 2

    def _estimate_enemy_escape_routes(self) -> int:
        """
        Roughly estimate how many exits the opponent has so we can stop
        overspending turds once they are already confined.
        """
        if not self.enemy_loc:
            return 4
        
        escapes = 0
        ex, ey = self.enemy_loc
        for d in Direction:
            nx, ny = self._apply_dir((ex, ey), d)
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                continue
            if (nx, ny) in self.confirmed_traps:
                continue
            if (nx, ny) in self.my_turd_positions:
                continue
            escapes += 1
        return escapes
    
    def _is_in_safe_zone(self, x, y) -> bool:
        safe_neighbors = self._count_safe_neighbors(x, y)
        return safe_neighbors >= 4

    def _get_quadrant(self, x, y) -> int:
        mid = self.board_size // 2
        if x < mid and y < mid:
            return 0
        elif x >= mid and y < mid:
            return 1
        elif x < mid and y >= mid:
            return 2
        else:
            return 3
    
    def _count_turds_in_quadrant(self, quadrant: int) -> int:
        count = 0
        for tx, ty in self.my_turd_positions:
            if self._get_quadrant(tx, ty) == quadrant:
                count += 1
        return count
    
    def _count_nearby_turds(self, x, y, radius=2) -> int:
        count = 0
        for tx, ty in self.my_turd_positions:
            if abs(tx - x) <= radius and abs(ty - y) <= radius:
                count += 1
        return count

    def _score_turd_placement(self, cx, cy) -> float:
        """
        RING v3 (Black Chicken mode): prioritize interior lanes and predictive blocks.
        """
        score = -40  # Was -50
        
        # 1. TIMING: Start EARLIER
        if self.turn_index < 12:  # Was 14
            return -500  # Too early
        elif self.turn_index < 18:  # Was 20
            score += 40  # Was 20 - Start considering earlier
        elif self.turn_index < 28:  # Was 30
            score += 100  # Was 80 - Good window
        else:
            score += 80  # Was 60 - Late game
        
        # 2. LAYER / LANE PRIORITY: strongly avoid edges, hug inner lanes
        layer = self._get_layer(cx, cy)
        lane_rank = self._lane_rank(cx, cy)
        
        if layer == 0:
            if self.board_control < 0.88:
                return -500  # absolutely never place edge turds early
            score -= 120
        elif lane_rank >= 0.9:
            score += 260
        elif lane_rank >= 0.35:
            score += 140
        else:
            score += 40
        
        if lane_rank >= 0.9:
            score += 80  # extra nudge for exact lane tiles
        
        # 3. NO CLUSTERING
        nearby_turds = self._count_nearby_turds(cx, cy, radius=2)
        if nearby_turds >= 2:
            score -= 350
        elif nearby_turds == 1:
            score -= 150
        
        # 4. QUADRANT BALANCE
        quadrant = self._get_quadrant(cx, cy)
        turds_in_quadrant = self._count_turds_in_quadrant(quadrant)
        
        if turds_in_quadrant == 0:
            score += 120
        elif turds_in_quadrant == 1:
            score -= 30
        else:
            score -= 100

        if self._forms_lane_barrier(cx, cy):
            score += 220  # completing an inner barrier is huge
        
        # 4b. Ease off turds once the opponent is already constrained
        enemy_escape_routes = self._estimate_enemy_escape_routes()
        if self.board_control >= 0.45:
            if enemy_escape_routes <= 1:
                score -= 200  # Already trapped, save turds for eggs
            elif enemy_escape_routes == 2:
                score -= 80
        
        # 5. BLOCK / FUNNEL THE OPPONENT
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            
            # Bonus if turd is between opponent and an edge
            # Check all 4 edges
            edges_blocked = 0
            
            # Top edge (y=0)
            if cy < oy and cy <= 2:
                edges_blocked += 1
            # Bottom edge (y=max)
            if cy > oy and cy >= self.board_size - 3:
                edges_blocked += 1
            # Left edge (x=0)
            if cx < ox and cx <= 2:
                edges_blocked += 1
            # Right edge (x=max)
            if cx > ox and cx >= self.board_size - 3:
                edges_blocked += 1
            
            score += edges_blocked * 80
            
            # Distance to opponent
            dist = abs(cx - ox) + abs(cy - oy)
            if dist == 2:
                score += 100
            elif dist == 3:
                score += 70
            elif dist == 4:
                score += 40
            
            # Stand between opponent and center lanes (black chicken funnel)
            center_line = (self.board_size - 1) / 2.0
            closer_to_center_x = abs(cx - center_line) < abs(ox - center_line)
            closer_to_center_y = abs(cy - center_line) < abs(oy - center_line)
            if lane_rank >= 0.9 and (closer_to_center_x or closer_to_center_y):
                score += 160
            
            # Predict enemy's next two center-seeking steps and reward blocking them
            step_dx = -1 if ox > center_line else (1 if ox < center_line else 0)
            step_dy = -1 if oy > center_line else (1 if oy < center_line else 0)
            px, py = ox, oy
            for _ in range(2):
                px += step_dx
                py += step_dy
                if (px, py) == (cx, cy):
                    score += 170
                    break
        
        # 6. OPPONENT'S EGG PARITY
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (cx + cy) % 2 == opp_parity:
                score += 80
        
        # 7. DON'T BLOCK OURSELVES
        if (cx, cy) in self.visited and len(self.visited) < 25:
            score -= 40

        if self.board_control >= 0.55 and self.turn_index > 30:
            score -= 60  # Late game: shift focus to eggs
        
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
        
        if self._is_layer_0(nx, ny):
            score += 60  # Edge eggs are valuable
        
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
        """
        RING v3: Inside-out control with Apex-level safety.
        
        Goals:
        - Match Apex's 0.05 trap thresholds
        - Keep layer-first exploration, but only when tiles are trusted
        - Add board-control awareness and a safe late-game push
        """
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
        safe_ratio = len(self.confirmed_safe) / self.total_tiles if self.total_tiles else 0.0
        
        # Phase detection (Apex-style thresholds + layer phases)
        very_late_phase = self.turn_index > 34
        late_phase = self.turn_index > 28
        mid_layer_phase = 16 < self.turn_index <= 28
        early_layer_phase = not mid_layer_phase and not late_phase
        
        if very_late_phase:
            RISK_VERY_SAFE = 0.08
            RISK_SAFE = 0.11
            RISK_ACCEPTABLE = 0.15
            RISK_CAUTION = 0.19
            CONFIDENCE_HIGH = 0.40
            CONFIDENCE_GOOD = 0.30
            CONFIDENCE_OK = 0.22
        elif late_phase:
            RISK_VERY_SAFE = 0.055
            RISK_SAFE = 0.08
            RISK_ACCEPTABLE = 0.11
            RISK_CAUTION = 0.15
            CONFIDENCE_HIGH = 0.48
            CONFIDENCE_GOOD = 0.38
            CONFIDENCE_OK = 0.28
        else:
            # Early + mid = strict like Apex
            RISK_VERY_SAFE = 0.045
            RISK_SAFE = 0.065
            RISK_ACCEPTABLE = 0.09
            RISK_CAUTION = 0.12
            CONFIDENCE_HIGH = 0.55
            CONFIDENCE_GOOD = 0.42
            CONFIDENCE_OK = 0.32
        
        in_safe_zone = self._is_in_safe_zone(nx, ny)
        safe_neighbors = self._count_safe_neighbors(nx, ny)
        evidence = self.cumulative_evidence[ny][nx]
        observations = self.observation_count[ny][nx]
        
        # Dynamic bonuses
        late_bonus = 1.32 if very_late_phase else (1.18 if late_phase else (1.08 if mid_layer_phase else 1.0))
        aggression_bonus = 1.25 if self.board_control < 0.5 else (1.10 if self.board_control < 0.65 else 1.0)
        late_push_ready = late_phase and safe_ratio >= 0.65
        push_multiplier = 1.12 if late_push_ready else 1.0
        risk_relief = 0.8 if late_phase else 1.0
        if late_push_ready:
            risk_relief *= 0.8
        
        # Get layer / lane info for target tile
        target_layer = self._get_layer(nx, ny)
        lane_rank = self._lane_rank(nx, ny)
        
        # BASE SAFETY SCORING (Apex thresholds, Ring bonuses)
        if is_confirmed_safe:
            if not is_visited:
                util += int(2300 * late_bonus * aggression_bonus * push_multiplier)
            else:
                util += 18
                
        elif risk < RISK_VERY_SAFE and confidence >= CONFIDENCE_HIGH:
            if not is_visited:
                util += int(1800 * late_bonus * aggression_bonus * push_multiplier)
            else:
                util += 14
                
        elif risk < RISK_SAFE and confidence >= CONFIDENCE_GOOD:
            if not is_visited:
                util += int(1350 * late_bonus * aggression_bonus)
            else:
                util += 10
                
        elif risk < RISK_ACCEPTABLE and confidence >= CONFIDENCE_OK and safe_neighbors >= 2:
            if not is_visited:
                util += int(950 * late_bonus)
            else:
                util += 7
                
        elif risk < RISK_CAUTION and in_safe_zone and evidence > 6:
            if not is_visited:
                util += int(650 * late_bonus)
            else:
                util += 5
                
        elif risk < RISK_CAUTION and safe_neighbors >= 3 and observations >= 2:
            if not is_visited:
                util += int(450 * late_bonus)
            else:
                util += 4
                
        else:
            risk_penalty = -520 - (risk * 2200)
            util += risk_penalty * risk_relief
        
        # Evidence bonuses - still reward cumulative info
        if not is_confirmed_safe:
            if evidence > 24:
                util += 260
            elif evidence > 15:
                util += 160
            elif evidence > 8:
                util += 90
            elif evidence < -10:
                util -= 140

        # Layer strategy – only when the tile is trusted
        layer_trusted = is_confirmed_safe or (risk < RISK_VERY_SAFE and confidence >= CONFIDENCE_HIGH)
        if not is_visited:
            if layer_trusted:
                if early_layer_phase:
                    if target_layer == 1:
                        util += 300
                    elif target_layer == 2:
                        util += 230
                    elif target_layer == 0:
                        util -= 70
                    else:
                        util += 80
                elif mid_layer_phase:
                    if target_layer == 1:
                        util += 200
                    elif target_layer == 2:
                        util += 150
                    elif target_layer == 0:
                        util += 60
                    else:
                        util += 50
                else:
                    if target_layer == 0:
                        util += 360
                    elif target_layer == 1:
                        util += 150
                    elif target_layer == 2:
                        util += 110
                    else:
                        util += 60
            elif early_layer_phase and target_layer == 0:
                util -= 60  # Unknown edges are still risky early

        # EXTRA lane preference (black chicken sweep)
        if not is_visited and layer_trusted and lane_rank > 0:
            util += int(90 * (1 + lane_rank * 2.2))

        # Heavy penalty for hugging edges before late game
        if not late_phase and target_layer == 0 and not is_visited:
            util -= 140

        # Egg bonuses (with late push)
        if mt == MoveType.EGG:
            egg_base = 900
            if late_push_ready:
                egg_base += 120
            util += egg_base
            
            # Corner eggs always valuable
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 430
            elif target_layer == 0:
                util += 170  # Edge eggs valuable
            elif target_layer == 1:
                util += 120  # Layer 1 eggs
            
            blocking_score = self._score_blocking_placement(nx, ny)
            util += blocking_score
            
            # Competitive bonus
            egg_diff = self.my_eggs - self.enemy_eggs
            if egg_diff < -3:
                util += 140
            elif egg_diff < -1:
                util += 90
            elif egg_diff < 0:
                util += 50

        # Direction momentum
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 30
            elif d == self._opposite_dir(self.last_direction):
                util -= 35
            else:
                # encourage the black chicken zig-zag sweep (alternate axes)
                horiz = {Direction.LEFT, Direction.RIGHT}
                vert = {Direction.UP, Direction.DOWN}
                if (self.last_direction in horiz and d in vert) or (self.last_direction in vert and d in horiz):
                    util += 25

        # Information gathering - only for trusted tiles
        if is_confirmed_safe and not is_visited:
            unknown_nearby = 0
            for ddy in range(-2, 3):
                for ddx in range(-2, 3):
                    tx, ty = nx + ddx, ny + ddy
                    if 0 <= tx < self.board_size and 0 <= ty < self.board_size:
                        if self.observation_count[ty][tx] < 3:
                            unknown_nearby += 1
            util += unknown_nearby * 15

        # Egg parity preference
        if self.my_egg_parity is not None:
            if (nx + ny) % 2 == self.my_egg_parity and not is_visited:
                util += 80

        # Turd placement
        if mt == MoveType.TURD:
            turd_score = self._score_turd_placement(cx, cy)
            util += turd_score
        
        # Territory expansion (only when safe)
        if self.enemy_loc and not is_visited and (is_confirmed_safe or (risk < RISK_VERY_SAFE and confidence >= CONFIDENCE_HIGH)):
            ex, ey = self.enemy_loc
            dist_to_enemy = abs(nx - ex) + abs(ny - ey)
            if dist_to_enemy > 4:
                util += 55
            
            center_line = (self.board_size - 1) / 2.0
            closer_to_center_x = abs(nx - center_line) < abs(ex - center_line)
            closer_to_center_y = abs(ny - center_line) < abs(ey - center_line)
            if lane_rank >= 0.35 and (closer_to_center_x or closer_to_center_y):
                util += 70  # keep ourselves between the opponent and the center lanes

        # Competitive exploration boost (safe tiles only)
        if is_confirmed_safe and not is_visited:
            egg_diff = self.my_eggs - self.enemy_eggs
            if egg_diff < -2:
                util += 90
            elif egg_diff < 0:
                util += 45

        if late_push_ready and not is_visited and layer_trusted:
            util += 110  # Encourage safe late-game surge

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
            thresh_safe = (0.11, 0.30)
            thresh_acceptable = (0.15, 0.22)
            thresh_caution = 0.19
        elif late_game:
            thresh_very_safe = (0.055, 0.48)
            thresh_safe = (0.08, 0.38)
            thresh_acceptable = (0.11, 0.28)
            thresh_caution = 0.15
        else:
            thresh_very_safe = (0.045, 0.55)
            thresh_safe = (0.065, 0.42)
            thresh_acceptable = (0.09, 0.32)
            thresh_caution = 0.12
        
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
                elif risk < thresh_caution and in_safe_zone and evidence > 6:
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
        
        if chosen[1] == MoveType.TURD:
            self.my_turd_positions.add((cx, cy))
        
        prev_enemy_eggs = len(self.enemy_egg_positions)
        if self.enemy_eggs > prev_enemy_eggs and self.enemy_loc:
            self.enemy_egg_positions.add(self.enemy_loc)

        self.last_position = (cx, cy)
        self.last_direction = chosen[0]
        self.intended_position = self._apply_dir((cx, cy), chosen[0])
        
        return chosen

