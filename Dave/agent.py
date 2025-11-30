from collections.abc import Callable
from typing import List, Tuple, Optional, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – DAVE (Aggressive Board Control + Barrier Partition)
# ================================================================
class PlayerAgent:
    """
    DAVE: Mimics David (black chicken) strategy.
    
    Philosophy:
    1. AGGRESSIVE early exploration - control 70%+ of board
    2. Take calculated risks with proper Bayesian beliefs
    3. THEN place barrier turds to confine opponent
    4. Barriers work because WE control the board
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
        
        # DAVE SPECIFIC: Track board control
        self.total_tiles = self.board_size * self.board_size
        self.board_control = 0.0  # Percentage of board we've explored

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
        
        # Update board control
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
                    
                    # DAVE: Faster confidence building for aggression
                    evidence_boost = 15 if dist_type == "adjacent" else 10 if dist_type == "diagonal" else 6
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
        obs_confidence = min(1.0, obs / 3.0)  # DAVE: Faster confidence (was 4.0)
        
        evidence = self.cumulative_evidence[y][x]
        if evidence > 15:  # Lower threshold for aggression
            obs_confidence = min(1.0, obs_confidence + 0.35)
        elif evidence > 8:
            obs_confidence = min(1.0, obs_confidence + 0.2)
        
        history = self.belief_history[y][x]
        if len(history) >= 2:  # Faster (was 3)
            avg_belief = sum(history) / len(history)
            if avg_belief < 0.06:
                obs_confidence = min(1.0, obs_confidence + 0.25)
        
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
        return safe_neighbors >= 3  # DAVE: Lower threshold (was 4)

    def _score_turd_placement(self, cx, cy) -> float:
        """
        DAVE: Barrier turds for CONFINEMENT after board control.
        
        Only place barriers when we control 50%+ of board.
        Focus on cutting opponent off from unexplored areas.
        """
        score = -80
        
        # 1. TIMING: Wait for board control
        if self.turn_index < 20:
            return -500  # Don't place early - focus on exploration
        
        if self.board_control < 0.40:
            return -400  # Don't have enough control yet
        
        # 2. CONFINEMENT BONUS: Place barriers to trap opponent
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            
            # Calculate opponent's "territory" (unexplored tiles near them)
            opp_territory = 0
            our_territory = 0
            
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if (x, y) not in self.visited and (x, y) not in self.opponent_visited:
                        dist_to_opp = abs(x - ox) + abs(y - oy)
                        if dist_to_opp <= 4:
                            opp_territory += 1
                        else:
                            our_territory += 1
            
            # Place turd to cut off opponent's territory
            turd_dist_to_opp = abs(cx - ox) + abs(cy - oy)
            
            if turd_dist_to_opp <= 3 and opp_territory > 5:
                score += 250  # Cut off their escape route!
            elif turd_dist_to_opp <= 5:
                score += 150
        
        # 3. BARRIER CREATION: Connect to edges/obstacles
        connects_to_edge = (cx == 0 or cx == self.board_size - 1 or 
                           cy == 0 or cy == self.board_size - 1)
        
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
        
        if adjacent_obstacles >= 2:
            score += 180
        elif adjacent_obstacles >= 1:
            score += 100
        
        if connects_to_edge:
            score += 150
        
        # 4. PARTITION: Cut board in half with opponent on smaller side
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            
            # Horizontal partition
            if cy != oy:
                tiles_opp_side = 0
                tiles_our_side = 0
                for y in range(self.board_size):
                    for x in range(self.board_size):
                        if (x, y) not in self.visited:
                            if (cy < oy and y < cy) or (cy > oy and y > cy):
                                tiles_our_side += 1
                            else:
                                tiles_opp_side += 1
                
                if tiles_opp_side < tiles_our_side and tiles_opp_side < 15:
                    score += 200  # Trapping opponent in smaller area!
            
            # Vertical partition
            if cx != ox:
                tiles_opp_side = 0
                tiles_our_side = 0
                for y in range(self.board_size):
                    for x in range(self.board_size):
                        if (x, y) not in self.visited:
                            if (cx < ox and x < cx) or (cx > ox and x > cx):
                                tiles_our_side += 1
                            else:
                                tiles_opp_side += 1
                
                if tiles_opp_side < tiles_our_side and tiles_opp_side < 15:
                    score += 200
        
        # 5. OPPONENT'S EGG PARITY
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (cx + cy) % 2 == opp_parity:
                score += 120
        
        # 6. DON'T BLOCK OURSELVES
        if (cx, cy) in self.visited and len(self.visited) < 30:
            score -= 60
        
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
        """
        DAVE: AGGRESSIVE exploration scoring.
        
        Key difference: Much higher bonuses for NEW tiles.
        Goal: Control 70%+ of board before placing barriers.
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
        
        # DAVE: More aggressive thresholds
        late_game = self.turn_index > 30
        very_late_game = self.turn_index > 36
        
        # AGGRESSIVE THRESHOLDS - willing to take more risks
        if very_late_game:
            RISK_VERY_SAFE = 0.10
            RISK_SAFE = 0.14
            RISK_ACCEPTABLE = 0.18
            RISK_CAUTION = 0.22
            CONFIDENCE_HIGH = 0.35
            CONFIDENCE_GOOD = 0.25
            CONFIDENCE_OK = 0.18
        elif late_game:
            RISK_VERY_SAFE = 0.08
            RISK_SAFE = 0.11
            RISK_ACCEPTABLE = 0.15
            RISK_CAUTION = 0.19
            CONFIDENCE_HIGH = 0.40
            CONFIDENCE_GOOD = 0.30
            CONFIDENCE_OK = 0.22
        else:
            # DAVE: More aggressive early/mid game
            RISK_VERY_SAFE = 0.06
            RISK_SAFE = 0.09
            RISK_ACCEPTABLE = 0.12
            RISK_CAUTION = 0.16
            CONFIDENCE_HIGH = 0.45
            CONFIDENCE_GOOD = 0.35
            CONFIDENCE_OK = 0.25
        
        in_safe_zone = self._is_in_safe_zone(nx, ny)
        safe_neighbors = self._count_safe_neighbors(nx, ny)
        evidence = self.cumulative_evidence[ny][nx]
        observations = self.observation_count[ny][nx]
        
        late_bonus = 1.3 if very_late_game else (1.15 if late_game else 1.0)
        
        # DAVE: MUCH HIGHER exploration bonuses
        aggression_bonus = 1.3 if self.board_control < 0.5 else 1.0  # Push hard until 50% control
        
        if is_confirmed_safe:
            if not is_visited:
                util += int(2500 * late_bonus * aggression_bonus)  # VERY HIGH (was 1800)
            else:
                util += 15
                
        elif risk < RISK_VERY_SAFE and confidence >= CONFIDENCE_HIGH:
            if not is_visited:
                util += int(2000 * late_bonus * aggression_bonus)  # HIGH (was 1400)
            else:
                util += 12
                
        elif risk < RISK_SAFE and confidence >= CONFIDENCE_GOOD:
            if not is_visited:
                util += int(1600 * late_bonus * aggression_bonus)  # HIGH (was 1000)
            else:
                util += 10
                
        elif risk < RISK_ACCEPTABLE and confidence >= CONFIDENCE_OK and safe_neighbors >= 1:  # Lowered from 2
            if not is_visited:
                util += int(1200 * late_bonus)  # (was 700)
            else:
                util += 8
                
        elif risk < RISK_CAUTION and in_safe_zone and evidence > 3:  # Lowered from 5
            if not is_visited:
                util += int(800 * late_bonus)  # (was 400)
            else:
                util += 5
                
        elif risk < RISK_CAUTION and safe_neighbors >= 2 and observations >= 1:  # Lowered requirements
            if not is_visited:
                util += int(600 * late_bonus)  # (was 300)
            else:
                util += 5
        
        # DAVE: Additional tier for moderate-risk expansion
        elif risk < RISK_CAUTION and evidence > 0:
            if not is_visited:
                util += int(400 * late_bonus)  # Take calculated risks!
            else:
                util += 3
                
        else:
            risk_penalty = -400 - (risk * 1500)  # Lower penalty (was -500, 2000)
            if late_game:
                risk_penalty = risk_penalty * 0.6
            util += risk_penalty
        
        # Evidence bonuses
        if not is_confirmed_safe:
            if evidence > 20:
                util += 250
            elif evidence > 12:
                util += 150
            elif evidence > 6:
                util += 80
            elif evidence < -10:
                util -= 150

        # Egg bonuses
        if mt == MoveType.EGG:
            util += 1000
            
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 500
            elif nx == 0 or nx == self.board_size - 1 or ny == 0 or ny == self.board_size - 1:
                util += 150
            
            blocking_score = self._score_blocking_placement(nx, ny)
            util += blocking_score
            
            egg_diff = self.my_eggs - self.enemy_eggs
            if egg_diff < -2:
                util += 200
            elif egg_diff < 0:
                util += 100

        # DAVE: Shorter edge preference - quickly move to center
        if self.turn_index <= 10:
            if self._is_edge(nx, ny) and not is_visited:
                util += 100
            elif not self._is_edge(nx, ny) and risk > 0.08:
                util -= 80  # Lower penalty (was -150)
        # After turn 10, push into center aggressively
        elif self.turn_index <= 20 and not is_visited:
            util += 150  # Bonus for ANY new tile

        # Direction momentum
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 35  # Higher (was 25)
            elif d == self._opposite_dir(self.last_direction):
                util -= 30

        # Information gathering
        if is_confirmed_safe and not is_visited:
            unknown_nearby = 0
            for ddy in range(-2, 3):
                for ddx in range(-2, 3):
                    tx, ty = nx + ddx, ny + ddy
                    if 0 <= tx < self.board_size and 0 <= ty < self.board_size:
                        if self.observation_count[ty][tx] < 2:  # Lower threshold
                            unknown_nearby += 1
            util += unknown_nearby * 20  # Higher (was 10)

        # Egg parity preference
        if self.my_egg_parity is not None:
            if (nx + ny) % 2 == self.my_egg_parity and not is_visited:
                util += 100

        # Turd placement
        if mt == MoveType.TURD:
            turd_score = self._score_turd_placement(cx, cy)
            util += turd_score
        
        # DAVE: Bonus for expanding our territory (away from opponent)
        if self.enemy_loc and not is_visited:
            ex, ey = self.enemy_loc
            dist_to_enemy = abs(nx - ex) + abs(ny - ey)
            if dist_to_enemy > 4:
                util += 80  # Claim territory far from opponent

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
        moderate_explore = []  # DAVE: New tier
        safe_revisit = []
        risky = []
        
        late_game = self.turn_index > 30
        very_late_game = self.turn_index > 36
        
        # AGGRESSIVE thresholds
        if very_late_game:
            thresh_very_safe = (0.10, 0.35)
            thresh_safe = (0.14, 0.25)
            thresh_acceptable = (0.18, 0.18)
            thresh_caution = 0.22
        elif late_game:
            thresh_very_safe = (0.08, 0.40)
            thresh_safe = (0.11, 0.30)
            thresh_acceptable = (0.15, 0.22)
            thresh_caution = 0.19
        else:
            thresh_very_safe = (0.06, 0.45)
            thresh_safe = (0.09, 0.35)
            thresh_acceptable = (0.12, 0.25)
            thresh_caution = 0.16
        
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
                elif risk < thresh_acceptable[0] and confidence >= thresh_acceptable[1] and safe_neighbors >= 1:
                    cautious_explore.append(m)
                elif risk < thresh_caution and in_safe_zone and evidence > 3:
                    cautious_explore.append(m)
                elif risk < thresh_caution and safe_neighbors >= 2 and observations >= 1:
                    cautious_explore.append(m)
                elif risk < thresh_caution and evidence > 0:
                    moderate_explore.append(m)  # DAVE: New tier for expansion
                else:
                    risky.append(m)

        # DAVE: Priority order with moderate exploration tier
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

        # DAVE: Take moderate risks for board control
        if moderate_explore and self.board_control < 0.5:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in moderate_explore]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > 0:  # Only if positive score
                return scored[0][1]

        if safe_revisit:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_revisit]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        if risky:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in risky]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > -200:
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

