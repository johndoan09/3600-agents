from collections.abc import Callable
from typing import List, Tuple, Optional, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – VOID (Proper Bayesian Trapdoor Belief System)
# ================================================================
class PlayerAgent:
    """
    VOID: Uses CORRECT Bayesian probability updates based on actual game rules.
    
    KEY INSIGHT: Silence does NOT mean safe!
    - Adjacent to trap: 35% chance of silence
    - Diagonal to trap: 64% chance of silence
    - 2 away from trap: 90% chance of silence
    
    Must accumulate MULTIPLE readings to be confident.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.my_egg_parity: Optional[int] = None

        # PROPER BAYESIAN BELIEFS: probability that each tile has a trap
        # Initialize with center-biased priors
        self.trap_belief = np.zeros((self.board_size, self.board_size))
        self._init_prior_beliefs()
        
        # MEMORY: Track belief history for each tile
        # This helps detect consistent low-risk areas
        self.belief_history = [[[] for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # Track number of observations for each tile (more = more confident)
        self.observation_count = np.zeros((self.board_size, self.board_size))
        
        # Track cumulative evidence (positive = safe evidence, negative = danger evidence)
        self.cumulative_evidence = np.zeros((self.board_size, self.board_size))
        
        # Confirmed traps and safe tiles
        self.confirmed_traps: Set[Tuple[int, int]] = set()
        self.confirmed_safe: Set[Tuple[int, int]] = set()  # Visited tiles only!

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
        """Initialize with center-biased priors (traps more likely in center)"""
        center = self.board_size / 2 - 0.5
        for y in range(self.board_size):
            for x in range(self.board_size):
                dist_to_center = abs(x - center) + abs(y - center)
                # Higher prior in center, lower at edges
                if dist_to_center <= 2:
                    self.trap_belief[y][x] = 0.15  # 15% prior in center
                elif dist_to_center <= 4:
                    self.trap_belief[y][x] = 0.08  # 8% in middle ring
                else:
                    self.trap_belief[y][x] = 0.02  # 2% at edges

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
        """Get distance type for probability calculations"""
        adx, ady = abs(dx), abs(dy)
        if adx + ady == 1:  # Adjacent (shares edge)
            return "adjacent"
        elif adx == 1 and ady == 1:  # Diagonal
            return "diagonal"
        elif adx + ady == 2:  # 2 away orthogonally
            return "two_away"
        return "far"

    def _get_signal_probability(self, dist_type: str) -> Tuple[float, float]:
        """
        Return (P(hear), P(feel)) based on distance type
        From game rules:
        - Adjacent: 50% hear, 30% feel
        - Diagonal: 25% hear, 15% feel
        - 2 away: 10% hear, 0% feel
        """
        if dist_type == "adjacent":
            return 0.50, 0.30
        elif dist_type == "diagonal":
            return 0.25, 0.15
        elif dist_type == "two_away":
            return 0.10, 0.00
        return 0.00, 0.00

    def _get_silence_probability(self, dist_type: str) -> float:
        """
        Return P(silence | trap at this distance)
        Silence = NOT heard AND NOT felt
        """
        p_hear, p_feel = self._get_signal_probability(dist_type)
        return (1 - p_hear) * (1 - p_feel)

    def _record_trap(self, loc):
        """Record a confirmed trap location"""
        x, y = loc
        self.confirmed_traps.add(loc)
        self.trap_belief[y][x] = 1.0  # 100% certain
        
        # Update beliefs for nearby tiles (trap found, so they're slightly safer)
        # But NOT 100% safe - there's still another trap!

    def _bayesian_update(self, loc, sensors):
        """
        PROPER Bayesian update with MEMORY component.
        
        Tracks belief history and cumulative evidence for better inference.
        """
        (hw, fw), (hb, fb) = sensors
        lx, ly = loc
        
        # Current tile is CONFIRMED safe (we're standing on it!)
        self.trap_belief[ly][lx] = 0.0
        self.cumulative_evidence[ly][lx] = 100  # Very safe
        self.confirmed_safe.add((lx, ly))
        self.visited.add((lx, ly))
        
        heard = hw or hb
        felt = fw or fb
        
        # Update beliefs for all nearby tiles
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = lx + dx, ly + dy
                if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                    continue
                
                # Skip confirmed tiles
                if (nx, ny) in self.confirmed_safe or (nx, ny) in self.confirmed_traps:
                    continue
                if (nx, ny) in self.visited or (nx, ny) in self.opponent_visited:
                    continue
                
                dist_type = self._get_distance_type(dx, dy)
                if dist_type == "far":
                    continue
                
                # Current prior
                prior = self.trap_belief[ny][nx]
                
                # Get signal probabilities for this distance
                p_hear, p_feel = self._get_signal_probability(dist_type)
                
                if heard or felt:
                    # GOT A SIGNAL - increase belief there's a trap nearby
                    if felt:
                        p_signal_given_trap = 1 - (1 - p_hear) * (1 - p_feel)
                        # MEMORY: Strong negative evidence
                        self.cumulative_evidence[ny][nx] -= 30 if dist_type == "adjacent" else 15
                    else:
                        p_signal_given_trap = p_hear
                        # MEMORY: Moderate negative evidence
                        self.cumulative_evidence[ny][nx] -= 15 if dist_type == "adjacent" else 8
                    
                    p_signal_given_no_trap = 0.1
                    
                    # Bayes update
                    likelihood_ratio = p_signal_given_trap / max(p_signal_given_no_trap, 0.01)
                    posterior = (prior * likelihood_ratio) / (prior * likelihood_ratio + (1 - prior))
                    
                    self.trap_belief[ny][nx] = min(0.95, posterior)
                else:
                    # SILENCE - decrease belief
                    p_silence_given_trap = self._get_silence_probability(dist_type)
                    p_silence_given_no_trap = 1.0
                    
                    numerator = p_silence_given_trap * prior
                    denominator = p_silence_given_trap * prior + p_silence_given_no_trap * (1 - prior)
                    
                    if denominator > 0:
                        posterior = numerator / denominator
                        self.trap_belief[ny][nx] = posterior
                    
                    # MEMORY: Positive evidence (silence = safer)
                    # Boost evidence more to build confidence faster
                    evidence_boost = 12 if dist_type == "adjacent" else 8 if dist_type == "diagonal" else 5
                    self.cumulative_evidence[ny][nx] += evidence_boost
                
                # Track observation count
                self.observation_count[ny][nx] += 1
                
                # MEMORY: Store belief history
                self.belief_history[ny][nx].append(self.trap_belief[ny][nx])
                # Keep only last 10 observations
                if len(self.belief_history[ny][nx]) > 10:
                    self.belief_history[ny][nx].pop(0)

    def _get_trap_risk(self, x, y) -> float:
        """Get the current trap probability for a tile"""
        if (x, y) in self.confirmed_traps:
            return 1.0
        if (x, y) in self.confirmed_safe or (x, y) in self.visited or (x, y) in self.opponent_visited:
            return 0.0
        return self.trap_belief[y][x]

    def _get_confidence(self, x, y) -> float:
        """
        Get confidence level using MEMORY component.
        Combines observation count + cumulative evidence + belief history consistency.
        """
        if (x, y) in self.confirmed_safe or (x, y) in self.visited or (x, y) in self.opponent_visited:
            return 1.0
        if (x, y) in self.confirmed_traps:
            return 1.0
        
        # Base confidence from observations
        obs = self.observation_count[y][x]
        obs_confidence = min(1.0, obs / 4.0)  # 4 observations = full confidence (was 5)
        
        # Boost confidence if cumulative evidence is strongly positive
        evidence = self.cumulative_evidence[y][x]
        if evidence > 20:
            obs_confidence = min(1.0, obs_confidence + 0.3)
        elif evidence > 10:
            obs_confidence = min(1.0, obs_confidence + 0.15)
        
        # Check belief history consistency (if beliefs have been consistently low)
        history = self.belief_history[y][x]
        if len(history) >= 3:
            avg_belief = sum(history) / len(history)
            if avg_belief < 0.05:  # Consistently very low
                obs_confidence = min(1.0, obs_confidence + 0.2)
        
        return obs_confidence
    
    def _count_safe_neighbors(self, x, y) -> int:
        """Count how many adjacent tiles are confirmed safe"""
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
        """Check if tile is surrounded by safe tiles (safe to explore aggressively)"""
        safe_neighbors = self._count_safe_neighbors(x, y)
        return safe_neighbors >= 4  # At least 4 of 8 neighbors are safe

    def _is_safe_enough(self, x, y, threshold=0.05) -> bool:
        """
        Is this tile safe enough to step on?
        Requires LOW trap probability AND HIGH confidence
        """
        if (x, y) in self.confirmed_traps:
            return False
        if (x, y) in self.confirmed_safe or (x, y) in self.visited or (x, y) in self.opponent_visited:
            return True
        
        risk = self._get_trap_risk(x, y)
        confidence = self._get_confidence(x, y)
        
        # Need BOTH low risk AND high confidence
        return risk < threshold and confidence > 0.6

    def _score_turd_placement(self, cx, cy) -> float:
        """
        AGGRESSIVE turd placement tuned to beat Shadow.
        
        Shadow wins by eggs (93%), so we need to BLOCK more aggressively.
        Shadow hits 0.70 traps/game - exploit this by forcing bad positions.
        """
        score = -30  # Lower base penalty (more willing to place)
        
        # 1. TIMING: Start earlier against aggressive opponents
        if self.turn_index < 12:
            return -500  # Never very early
        elif self.turn_index < 20:
            score -= 50   # Slight discourage early-mid
        elif self.turn_index > 25:
            score += 80   # Encourage mid-late
        
        # 2. OPPONENT PROXIMITY: VERY aggressive when close
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            dist = abs(cx - ox) + abs(cy - oy)
            
            if dist == 1:
                score += 350  # Adjacent! MUST place
            elif dist == 2:
                score += 220  # Close, high priority
            elif dist <= 3:
                score += 130  # Nearby, good
            elif dist <= 5:
                score += 50   # Medium range
            else:
                score -= 80   # Too far
        
        # 3. BLOCK CORNERS: Enhanced corner blocking
        corners = [(0, 0), (0, self.board_size - 1), 
                   (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            for corner_x, corner_y in corners:
                dist_opp_to_corner = abs(ox - corner_x) + abs(oy - corner_y)
                dist_turd_to_corner = abs(cx - corner_x) + abs(cy - corner_y)
                
                if dist_turd_to_corner < dist_opp_to_corner and dist_turd_to_corner <= 4:
                    score += 150
                    if dist_opp_to_corner <= 5:
                        score += 80
        
        # 4. OPPONENT'S EGG PARITY: Strong blocking
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (cx + cy) % 2 == opp_parity:
                score += 150
        
        # 5. TRAP FORCING: Push opponent toward dangerous tiles
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            danger_tiles_nearby = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        risk = self._get_trap_risk(nx, ny)
                        if risk > 0.12:
                            danger_tiles_nearby += 1
                            if abs(cx - nx) <= 2 and abs(cy - ny) <= 2:
                                score += 80
            
            if danger_tiles_nearby >= 2:
                score += 60
        
        # 6. DON'T BLOCK OURSELVES
        if (cx, cy) in self.visited and len(self.visited) < 15:
            score -= 40
        
        # 7. EDGE TURDS: Good for cornering opponent
        if cx == 0 or cx == self.board_size - 1 or cy == 0 or cy == self.board_size - 1:
            score += 60
            if self.enemy_loc:
                ox, oy = self.enemy_loc
                if ox == 0 or ox == self.board_size - 1 or oy == 0 or oy == self.board_size - 1:
                    score += 50
        
        # 8. WINNING BONUS: More aggressive if behind on eggs
        egg_diff = self.my_eggs - self.enemy_eggs
        if egg_diff < -2:
            score += 50
        elif egg_diff < 0:
            score += 25
        
        return score

    def _score_blocking_placement(self, nx, ny) -> float:
        """
        Score egg placement based on how well it blocks the opponent.
        Key strategies:
        1. Block opponent's path to corners
        2. Create lines of eggs that block parity
        3. Place eggs near opponent to disrupt their plans
        """
        score = 0.0
        
        # 1. Block opponent's path to corners
        corners = [(0, 0), (0, self.board_size - 1), 
                   (self.board_size - 1, 0), (self.board_size - 1, self.board_size - 1)]
        
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            for cx, cy in corners:
                dist_opp_to_corner = abs(ox - cx) + abs(oy - cy)
                dist_egg_to_corner = abs(nx - cx) + abs(ny - cy)
                
                # If egg is between opponent and corner, and closer to corner
                if dist_egg_to_corner < dist_opp_to_corner and dist_egg_to_corner <= 3:
                    score += 100  # Strong bonus for blocking corner access
        
        # 2. Create lines of eggs (blocking opponent's parity)
        line_eggs = 0
        for offset in [-2, -1, 1, 2]:
            if (nx + offset, ny) in self.my_egg_positions:
                line_eggs += 1
            if (nx, ny + offset) in self.my_egg_positions:
                line_eggs += 1
        score += line_eggs * 50  # Bonus for extending lines
        
        # 3. Edge eggs create barriers
        if nx == 0 or nx == self.board_size - 1 or ny == 0 or ny == self.board_size - 1:
            score += 60  # Edge eggs are valuable for blocking
        
        # 4. Near opponent - disrupt their immediate plans
        if self.enemy_loc:
            dist_to_enemy = abs(nx - self.enemy_loc[0]) + abs(ny - self.enemy_loc[1])
            if dist_to_enemy <= 3:
                score += 80  # Disrupt nearby opponent
            elif dist_to_enemy <= 5:
                score += 40
        
        # 5. Block opponent's egg parity
        if self.my_egg_parity is not None:
            opp_parity = 1 - self.my_egg_parity
            if (nx + ny) % 2 == opp_parity:
                score += 70  # Placing on opponent's parity blocks them
        
        return score

    def _score_move(self, move, board_state, cx, cy) -> float:
        """
        Score moves for MAXIMUM EGG PRODUCTION with 0.05 traps/game target.
        
        Strategy: Only step on tiles we're HIGHLY confident are safe.
        Build confidence through multiple silence readings.
        """
        d, mt = move
        nx, ny = self._apply_dir((cx, cy), d)

        # HARD BAN on confirmed traps
        if (nx, ny) in self.confirmed_traps:
            return -1e12

        risk = self._get_trap_risk(nx, ny)
        confidence = self._get_confidence(nx, ny)
        is_visited = (nx, ny) in self.visited
        is_opponent_visited = (nx, ny) in self.opponent_visited
        is_confirmed_safe = (nx, ny) in self.confirmed_safe or is_visited or is_opponent_visited
        
        util = 0.0
        
        # ===== LATE GAME DETECTION =====
        # Last 12 turns (turn > 28 of 40) - we know more, can push harder
        late_game = self.turn_index > 28
        very_late_game = self.turn_index > 34  # Last 6 turns - maximum push
        
        # ===== DYNAMIC THRESHOLDS: Tighter early, relaxed late =====
        if very_late_game:
            # VERY LATE GAME: Maximum aggression, board is well-known
            RISK_VERY_SAFE = 0.08
            RISK_SAFE = 0.12
            RISK_ACCEPTABLE = 0.16
            RISK_CAUTION = 0.20
            CONFIDENCE_HIGH = 0.40
            CONFIDENCE_GOOD = 0.30
            CONFIDENCE_OK = 0.20
        elif late_game:
            # LATE GAME: More aggressive, we have good data
            RISK_VERY_SAFE = 0.06
            RISK_SAFE = 0.09
            RISK_ACCEPTABLE = 0.12
            RISK_CAUTION = 0.16
            CONFIDENCE_HIGH = 0.45
            CONFIDENCE_GOOD = 0.35
            CONFIDENCE_OK = 0.25
        else:
            # NORMAL GAME: Standard safe thresholds
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
        
        # ===== LATE GAME BONUS MULTIPLIER =====
        late_bonus = 1.3 if very_late_game else (1.15 if late_game else 1.0)
        
        # ===== TIERED SAFETY SCORING =====
        
        if is_confirmed_safe:
            if not is_visited:
                util += int(1800 * late_bonus)  # Push harder late game
            else:
                util += 20
                
        elif risk < RISK_VERY_SAFE and confidence >= CONFIDENCE_HIGH:
            if not is_visited:
                util += int(1400 * late_bonus)
            else:
                util += 15
                
        elif risk < RISK_SAFE and confidence >= CONFIDENCE_GOOD:
            if not is_visited:
                util += int(1000 * late_bonus)
            else:
                util += 10
                
        elif risk < RISK_ACCEPTABLE and confidence >= CONFIDENCE_OK and safe_neighbors >= 2:
            if not is_visited:
                util += int(700 * late_bonus)
            else:
                util += 8
                
        elif risk < RISK_CAUTION and in_safe_zone and evidence > 5:
            if not is_visited:
                util += int(400 * late_bonus)
            else:
                util += 5
                
        elif risk < RISK_CAUTION and safe_neighbors >= 3 and observations >= 2:
            if not is_visited:
                util += int(300 * late_bonus)
            else:
                util += 5
                
        else:
            # TOO RISKY - but less penalty in late game (we need to move!)
            risk_penalty = -500 - (risk * 2000)
            if late_game:
                risk_penalty = risk_penalty * 0.7  # 30% less penalty late
            util += risk_penalty
        
        # ===== CUMULATIVE EVIDENCE BONUS =====
        # Strong positive evidence = extra confidence
        if not is_confirmed_safe:
            if evidence > 25:
                util += 200  # Very strong safe evidence
            elif evidence > 15:
                util += 100  # Strong safe evidence
            elif evidence > 8:
                util += 50   # Good safe evidence
            elif evidence < -10:
                util -= 200  # Danger signals accumulated

        # ===== EGG PRIORITY WITH BLOCKING STRATEGY =====
        if mt == MoveType.EGG:
            util += 800  # Base egg bonus
            
            # Corner eggs - very valuable
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 400
            
            # BLOCKING STRATEGY: Score eggs that block opponent
            blocking_score = self._score_blocking_placement(nx, ny)
            util += blocking_score

        # ===== EDGE PREFERENCE (early game) =====
        if self.turn_index <= 20:
            if self._is_edge(nx, ny) and not is_visited:
                util += 100
            elif not self._is_edge(nx, ny) and risk > 0.05:
                util -= 150  # Penalize risky center moves early

        # ===== DIRECTION MOMENTUM =====
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 25
            elif d == self._opposite_dir(self.last_direction):
                util -= 40

        # ===== INFORMATION GATHERING =====
        # Prefer moves that let us observe unknown tiles (original v7 logic)
        if is_confirmed_safe and not is_visited:
            unknown_nearby = 0
            for ddy in range(-2, 3):
                for ddx in range(-2, 3):
                    tx, ty = nx + ddx, ny + ddy
                    if 0 <= tx < self.board_size and 0 <= ty < self.board_size:
                        if self.observation_count[ty][tx] < 3:
                            unknown_nearby += 1
            util += unknown_nearby * 10

        # ===== SMART TURD STRATEGY =====
        if mt == MoveType.TURD:
            turd_score = self._score_turd_placement(cx, cy)
            util += turd_score

        return util + np.random.random() * 0.01

    def _choose_move(self, board_state, sensors, time_left):
        """Choose move prioritizing CONFIRMED safe tiles"""
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()

        # Categorize moves - TUNED FOR MORE EGGS + LOW TRAPS
        safe_new = []           # Confirmed safe, not visited
        very_safe = []          # <4% risk, good confidence
        safe_explore = []       # <6% risk, decent confidence  
        cautious_explore = []   # <9% risk, ok confidence, neighbors
        safe_revisit = []       # Confirmed safe, visited
        risky = []              # Everything else
        
        # LATE GAME: Relax categorization thresholds
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
                continue  # Never
            
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
                
                # LATE-GAME AWARE categorization
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

        # STRICT PRIORITY ORDER FOR 0.05 TRAPS/GAME
        
        # 1. Safe new tiles - ALWAYS FIRST (confirmed safe, unvisited)
        if safe_new:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_new]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # 2. Very safe (<3% risk, high confidence)
        if very_safe:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in very_safe]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # 3. Safe explore (<5% risk, good confidence)
        if safe_explore:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_explore]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # 4. Cautious explore (<8% risk, surrounded by safe)
        if cautious_explore:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in cautious_explore]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # 5. Safe revisits - BEFORE risky moves!
        if safe_revisit:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_revisit]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # 6. Risky moves (last resort - avoid at all costs)
        if risky:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in risky]
            scored.sort(key=lambda x: x[0], reverse=True)
            # Only take risky move if score is above threshold
            if scored[0][0] > -300:
                return scored[0][1]
            # Otherwise prefer staying still if possible
            for m in safe_revisit:
                return m

        return moves[0]

    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        self.my_eggs = board_state.chicken_player.get_eggs_laid()
        self.enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        
        cx, cy = board_state.chicken_player.get_location()
        
        # Record spawn
        if self.spawn_position is None:
            self.spawn_position = (cx, cy)
        
        # Detect if we hit a trap
        if self.intended_position is not None and self.last_position is not None:
            intended = self.intended_position
            current = (cx, cy)
            
            if current == self.spawn_position and intended != self.spawn_position:
                self._record_trap(intended)
        
        # Track opponent (confirmed safe!)
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

        # Record found traps
        try:
            for loc in board_state.found_trapdoors:
                if tuple(loc) not in self.confirmed_traps:
                    self._record_trap(tuple(loc))
        except:
            pass
        
        # BAYESIAN UPDATE based on sensors
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

        # Track egg positions for blocking strategy
        if chosen[1] == MoveType.EGG:
            self.my_egg_positions.add((cx, cy))
        
        # Track enemy eggs (approximate from board state)
        prev_enemy_eggs = len(self.enemy_egg_positions)
        if self.enemy_eggs > prev_enemy_eggs and self.enemy_loc:
            self.enemy_egg_positions.add(self.enemy_loc)

        self.last_position = (cx, cy)
        self.last_direction = chosen[0]
        self.intended_position = self._apply_dir((cx, cy), chosen[0])
        
        return chosen



