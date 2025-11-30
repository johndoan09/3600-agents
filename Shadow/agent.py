from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – SHADOW (Black Chicken Strategy)
# ================================================================
class PlayerAgent:
    """
    SHADOW: Implements the "Black Chicken" strategy observed in competition.
    
    Core Philosophy: "Patient Sensor Accumulation, Confident Exploration"
    
    Key Strategies:
    1. EDGE-FIRST: Explore edges early to gather safe silence readings
    2. SILENCE EXPLOITATION: When silence, mark ALL 25 tiles as FULLY safe
    3. SIGNAL TRIANGULATION: Use multiple readings to pinpoint exact trap location
    4. CONFIDENT ADJACENCY: Once trap is located, walk right next to it safely
    5. PATIENCE: Don't rush center until you have enough safety data
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.my_egg_parity: Optional[int] = None

        # SAFETY MAP: 0 = unknown, positive = confidence level, -999 = confirmed trap
        self.safety_map = np.zeros((self.board_size, self.board_size))
        
        # SIGNAL ACCUMULATOR: Track how many danger signals point to each tile
        self.danger_signals = np.zeros((self.board_size, self.board_size))
        
        # Track positions where we received signals (for triangulation)
        self.signal_positions: List[Tuple[int, int, bool, bool]] = []  # (x, y, heard, felt)
        
        # Confirmed trap locations (pinpointed through triangulation)
        self.confirmed_traps: Set[Tuple[int, int]] = set()
        self.probable_traps: Set[Tuple[int, int]] = set()  # High confidence but not 100%

        # Visited tiles
        self.visited: Set[Tuple[int, int]] = set()
        
        # Silence zones - tiles confirmed safe by silence
        self.silence_confirmed: Set[Tuple[int, int]] = set()

        # Direction tracking
        self.last_direction: Optional[Direction] = None
        
        # Track opponent path (confirmed safe!)
        self.opponent_visited: Set[Tuple[int, int]] = set()

        self.turn_index = 0
        self.my_eggs = 0
        self.enemy_eggs = 0
        self.enemy_loc: Optional[Tuple[int, int]] = None
        
        # Phase tracking
        self.exploration_phase = True  # Start in careful exploration mode
        self.tiles_with_silence = 0  # Count of tiles where we got silence

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

    def _is_near_edge(self, x, y) -> bool:
        return x <= 1 or x >= self.board_size - 2 or y <= 1 or y >= self.board_size - 2

    def _distance_to_center(self, x, y) -> float:
        center = self.board_size / 2 - 0.5
        return abs(x - center) + abs(y - center)

    def _record_trap(self, loc):
        x, y = loc
        self.confirmed_traps.add(loc)
        self.safety_map[y][x] = -999
        self.danger_signals[y][x] = 100

    def _process_sensors(self, loc, sensors):
        """
        BLACK CHICKEN'S SECRET: Aggressive silence exploitation + triangulation
        """
        (hw, fw), (hb, fb) = sensors
        lx, ly = loc
        
        heard = hw or hb
        felt = fw or fb
        
        # Current tile is always safe
        self.safety_map[ly][lx] = 100
        self.visited.add((lx, ly))

        if not heard and not felt:
            # ========== SILENCE = GOLD ==========
            # ALL tiles within 2 are CONFIRMED SAFE!
            self.tiles_with_silence += 1
            
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        dist = abs(dx) + abs(dy)
                        if dist <= 2:
                            # FULLY SAFE - mark with high confidence
                            self.silence_confirmed.add((nx, ny))
                            self.safety_map[ny][nx] = max(self.safety_map[ny][nx], 100)
                            self.danger_signals[ny][nx] = 0  # Clear any suspicion
                            
                            # Remove from probable traps if it was there
                            if (nx, ny) in self.probable_traps:
                                self.probable_traps.discard((nx, ny))
        else:
            # ========== GOT A SIGNAL - TRIANGULATE ==========
            self.signal_positions.append((lx, ly, heard, felt))
            
            # Mark nearby tiles as suspicious
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        # Skip if already confirmed safe by silence
                        if (nx, ny) in self.silence_confirmed:
                            continue
                        if (nx, ny) in self.visited:
                            continue
                        if (nx, ny) in self.opponent_visited:
                            continue
                        
                        dist = abs(dx) + abs(dy)
                        if dist == 0:
                            continue
                        
                        if felt and dist == 1:
                            # FELT = trap is ADJACENT! Very dangerous
                            self.danger_signals[ny][nx] += 5
                            self.safety_map[ny][nx] -= 50
                            self.probable_traps.add((nx, ny))
                        elif felt and dist == 2:
                            self.danger_signals[ny][nx] += 2
                            self.safety_map[ny][nx] -= 20
                        elif heard and dist == 1:
                            self.danger_signals[ny][nx] += 3
                            self.safety_map[ny][nx] -= 30
                        elif heard and dist == 2:
                            self.danger_signals[ny][nx] += 1
                            self.safety_map[ny][nx] -= 10
            
            # Try to triangulate trap location
            self._triangulate_traps()

    def _triangulate_traps(self):
        """
        Use multiple signal readings to pinpoint exact trap locations.
        If we have multiple "felt" signals, the intersection is the trap.
        """
        if len(self.signal_positions) < 2:
            return
        
        # Find tiles that are adjacent to ALL felt positions
        felt_positions = [(x, y) for x, y, h, f in self.signal_positions if f]
        
        if len(felt_positions) >= 2:
            # Find intersection of adjacent tiles
            candidates = None
            for fx, fy in felt_positions:
                adjacent = set()
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = fx + dx, fy + dy
                        if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                            if (nx, ny) not in self.silence_confirmed:
                                if (nx, ny) not in self.visited:
                                    if (nx, ny) not in self.opponent_visited:
                                        adjacent.add((nx, ny))
                
                if candidates is None:
                    candidates = adjacent
                else:
                    candidates = candidates.intersection(adjacent)
            
            # If only 1-2 candidates remain, they're probably traps
            if candidates and len(candidates) <= 2:
                for c in candidates:
                    if self.danger_signals[c[1]][c[0]] >= 5:
                        self.probable_traps.add(c)
                        self.safety_map[c[1]][c[0]] = -500

    def _is_confirmed_safe(self, x, y) -> bool:
        """Check if a tile is DEFINITELY safe"""
        if (x, y) in self.confirmed_traps:
            return False
        if (x, y) in self.silence_confirmed:
            return True
        if (x, y) in self.visited:
            return True
        if (x, y) in self.opponent_visited:
            return True
        return False

    def _is_probably_dangerous(self, x, y) -> bool:
        """Check if a tile is probably a trap"""
        if (x, y) in self.confirmed_traps:
            return True
        if (x, y) in self.probable_traps:
            return True
        if self.danger_signals[y][x] >= 6:
            return True
        return False

    def _score_move(self, move, board_state, cx, cy) -> float:
        """
        BLACK CHICKEN SCORING:
        1. Never step on confirmed/probable traps
        2. Strongly prefer silence-confirmed safe tiles
        3. Edge preference in early game
        4. New tiles over revisits
        """
        d, mt = move
        nx, ny = self._apply_dir((cx, cy), d)

        # ===== HARD BANS =====
        if (nx, ny) in self.confirmed_traps:
            return -1e12
        
        if (nx, ny) in self.probable_traps:
            return -1e10
        
        if self._is_probably_dangerous(nx, ny):
            return -1e9

        util = 0.0
        
        is_new = (nx, ny) not in self.visited
        is_safe = self._is_confirmed_safe(nx, ny)
        is_edge = self._is_edge(nx, ny)
        is_near_edge = self._is_near_edge(nx, ny)
        danger = self.danger_signals[ny][nx]
        
        # ===== STRICT SAFETY SCORING =====
        if is_safe and is_new:
            # BEST: New tile that's confirmed safe - HUGE bonus
            util += 1000
        elif is_safe:
            # Good: Revisiting safe tile - still very good
            util += 500
        elif is_new and danger == 0:
            # Unknown but no danger signals - ONLY take if no safe option
            util -= 100  # Negative! We prefer safe revisits over unknown
        elif is_new and danger <= 2:
            # Some suspicion - really avoid
            util -= 500
        elif is_new:
            # High suspicion - nearly banned
            util -= 1000
        else:
            # Revisiting unknown tile
            util -= 200

        # ===== EDGE-FIRST EXPLORATION (Extended to turn 30) =====
        if self.turn_index <= 30:
            if is_edge and is_new and is_safe:
                util += 200  # Strong edge preference for SAFE edges
            elif is_near_edge and is_new and is_safe:
                util += 100
            elif is_edge and is_new and danger == 0:
                util += 50   # Edge unknown is better than center unknown
            elif not is_near_edge and not is_safe:
                # HEAVY penalty for center exploration when not confirmed safe
                util -= 500

        # ===== EGG PRIORITY =====
        if mt == MoveType.EGG:
            util += 400
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 300  # Corner eggs

        # ===== DIRECTION MOMENTUM =====
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 30
            elif d == self._opposite_dir(self.last_direction):
                util -= 50

        # ===== TURD STRATEGY =====
        if mt == MoveType.TURD:
            if self.turn_index < 20:
                util -= 500
            else:
                # Place on opponent's path or parity
                if self.my_egg_parity is not None:
                    opp_parity = 1 - self.my_egg_parity
                    if (cx + cy) % 2 == opp_parity:
                        util += 80
                if self.enemy_loc:
                    dist = abs(cx - self.enemy_loc[0]) + abs(cy - self.enemy_loc[1])
                    if dist <= 4:
                        util += 100
                util -= 20

        # ===== EXPLORATION BONUS FOR INFORMATION GATHERING =====
        # Prefer tiles that will give us MORE information
        if is_new and is_safe:
            # Count how many unknown tiles are adjacent
            unknown_adjacent = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ax, ay = nx + dx, ny + dy
                    if 0 <= ax < self.board_size and 0 <= ay < self.board_size:
                        if not self._is_confirmed_safe(ax, ay) and (ax, ay) not in self.visited:
                            unknown_adjacent += 1
            # More unknown neighbors = more potential information
            util += unknown_adjacent * 15

        return util + np.random.random() * 0.01

    def _choose_move(self, board_state, sensors, time_left):
        """
        BLACK CHICKEN'S STRICT RULE:
        NEVER step on unknown tiles if ANY safe option exists.
        Only step on unknown as ABSOLUTE last resort.
        """
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()

        # Separate moves by category - STRICT classification
        safe_new_moves = []      # Confirmed safe + not visited
        safe_revisit_moves = []  # Confirmed safe + already visited
        low_risk_moves = []      # Unknown but NO danger signals at all
        risky_moves = []         # Unknown with some danger signals
        dangerous_moves = []     # Probable traps
        
        for m in moves:
            d, mt = m
            nx, ny = self._apply_dir((cx, cy), d)
            
            if (nx, ny) in self.confirmed_traps or (nx, ny) in self.probable_traps:
                dangerous_moves.append(m)
            elif self._is_confirmed_safe(nx, ny):
                if (nx, ny) not in self.visited:
                    safe_new_moves.append(m)
                else:
                    safe_revisit_moves.append(m)
            elif self.danger_signals[ny][nx] == 0:
                # Unknown but ZERO danger signals - lowest risk unknown
                low_risk_moves.append(m)
            elif not self._is_probably_dangerous(nx, ny):
                risky_moves.append(m)
            else:
                dangerous_moves.append(m)

        # ========== STRICT PRIORITY ORDER ==========
        
        # PRIORITY 1: Safe new tiles - ALWAYS take if available
        if safe_new_moves:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_new_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # PRIORITY 2: Safe revisits - take over ANY unknown tile
        if safe_revisit_moves:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in safe_revisit_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # PRIORITY 3: ONLY if no safe options exist - low risk unknown (edge preferred)
        if low_risk_moves:
            # Prefer edges when forced to take unknown
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in low_risk_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            # Extra check: only take if on edge or near edge
            best_move = scored[0][1]
            nx, ny = self._apply_dir((cx, cy), best_move[0])
            if self._is_near_edge(nx, ny) or len(safe_new_moves) == 0 and len(safe_revisit_moves) == 0:
                return best_move
            # If not near edge, check if we have any edge option
            for score, m in scored:
                nx, ny = self._apply_dir((cx, cy), m[0])
                if self._is_edge(nx, ny):
                    return m
            # No edge option, take best low-risk
            return scored[0][1]

        # PRIORITY 4: Risky moves - ONLY as last resort before dangerous
        if risky_moves:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in risky_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # PRIORITY 5: Dangerous moves - absolute last resort
        if dangerous_moves:
            scored = [(self._score_move(m, board_state, cx, cy), m) for m in dangerous_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # Fallback
        return moves[0]

    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        self.my_eggs = board_state.chicken_player.get_eggs_laid()
        self.enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        
        # Track opponent (CONFIRMED SAFE TILES!)
        try:
            self.enemy_loc = board_state.chicken_enemy.get_location()
            if self.enemy_loc is not None:
                ex, ey = self.enemy_loc
                if (ex, ey) not in self.opponent_visited:
                    self.opponent_visited.add((ex, ey))
                    self.safety_map[ey][ex] = 100
                    self.silence_confirmed.add((ex, ey))
        except:
            self.enemy_loc = None

        # Record found traps
        try:
            for loc in board_state.found_trapdoors:
                if loc not in self.confirmed_traps:
                    self._record_trap(tuple(loc))
        except:
            pass

        cx, cy = board_state.chicken_player.get_location()
        
        # Process sensor data (THE KEY!)
        self._process_sensors((cx, cy), sensors)

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # Set egg parity
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

