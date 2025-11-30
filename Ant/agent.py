from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – ANT (Aggressive + Smart Trapdoor Detection)
# ================================================================
class PlayerAgent:
    """
    ANT: Aggressive explorer with precise trapdoor detection
    
    Key differences from Pirate:
    - TRUSTS sensor data strongly (silence = definitely safe)
    - Pinpoints trapdoor location precisely when signals detected
    - No edge preference - explores everywhere confidently
    - Maximum coverage priority
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.my_egg_parity: Optional[int] = None

        # SAFETY MAP: Positive = safe, Negative = dangerous, 0 = unknown
        # This is the KEY difference - we track safety more precisely
        self.safety_map = np.zeros((self.board_size, self.board_size))
        
        # Track signal history for each tile
        self.danger_signals = np.zeros((self.board_size, self.board_size))
        self.silence_signals = np.zeros((self.board_size, self.board_size))

        # Confirmed traps
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()

        # VISITED TILES
        self.visited: Set[Tuple[int, int]] = set()

        # Direction tracking
        self.last_direction: Optional[Direction] = None
        
        # Loop detection
        self.recent_positions: List[Tuple[int, int]] = []

        self.turn_index = 0
        self.my_eggs = 0
        self.enemy_eggs = 0

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
        """Mark a confirmed trapdoor"""
        self.confirmed_trapdoors.add(loc)
        x, y = loc
        self.safety_map[y][x] = -1000  # Definitely dangerous

    def _update_safety_from_sensors(self, loc, sensors):
        """
        KEY FUNCTION: Update safety map based on sensor readings
        
        Sensors format: ((heard_white, felt_white), (heard_black, felt_black))
        - heard: trapdoor within 2 tiles (manhattan distance)
        - felt: trapdoor within 1 tile (adjacent or diagonal)
        """
        (hw, fw), (hb, fb) = sensors
        cx, cy = loc
        
        any_signal = hw or fw or hb or fb
        felt_signal = fw or fb
        
        # Current tile is ALWAYS safe (we're standing on it)
        self.safety_map[cy][cx] = 100
        self.visited.add((cx, cy))
        
        if not any_signal:
            # === SILENCE: No trapdoor within 2 tiles! ===
            # This is VERY valuable information - mark all nearby tiles as SAFE
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        dist = abs(dx) + abs(dy)
                        if dist <= 2:
                            # Strong safety boost - we KNOW there's no trap here
                            self.silence_signals[ny][nx] += 1
                            if dist == 0:
                                self.safety_map[ny][nx] = 100
                            elif dist == 1:
                                self.safety_map[ny][nx] = max(self.safety_map[ny][nx], 80)
                            else:
                                self.safety_map[ny][nx] = max(self.safety_map[ny][nx], 50)
        else:
            # === SIGNAL DETECTED: Trapdoor nearby! ===
            # Mark potential locations as VERY dangerous
            
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        dist = abs(dx) + abs(dy)
                        
                        # Skip tiles we've already confirmed safe by standing on them
                        if (nx, ny) in self.visited:
                            continue
                        
                        if felt_signal and dist <= 1 and dist > 0:
                            # FELT signal - trapdoor is ADJACENT! Very dangerous!
                            self.danger_signals[ny][nx] += 5
                            self.safety_map[ny][nx] -= 200  # Much stronger penalty
                        elif any_signal and dist <= 2 and dist > 0:
                            # HEARD signal - trapdoor within 2 tiles
                            self.danger_signals[ny][nx] += 2
                            self.safety_map[ny][nx] -= 50

    def _is_stuck_in_loop(self) -> bool:
        """Detect if we're bouncing between same tiles"""
        if len(self.recent_positions) < 4:
            return False
        # Check if we're bouncing between 2-3 tiles
        recent = self.recent_positions[-6:]
        unique = set(recent)
        return len(unique) <= 2
    
    def _get_tile_safety(self, x, y) -> float:
        """Get safety score for a tile (higher = safer)"""
        if (x, y) in self.confirmed_trapdoors:
            return -1000
        if (x, y) in self.visited:
            return 100  # We've been there, definitely safe
        
        safety = self.safety_map[y][x]
        
        # Boost safety if we've had multiple silence signals covering this tile
        silence_boost = self.silence_signals[y][x] * 25
        
        # STRONG danger penalty - if we've detected danger here, AVOID IT
        danger_count = self.danger_signals[y][x]
        if danger_count >= 5:
            # Very high danger - almost certainly a trap
            danger_penalty = 500
        elif danger_count >= 3:
            # High danger
            danger_penalty = 200
        else:
            danger_penalty = danger_count * 40
        
        return safety + silence_boost - danger_penalty

    def _score_move(self, move, board_state, cx, cy, is_revisit: bool) -> float:
        """Score a move - aggressive but respects danger signals"""
        d, mt = move
        nx, ny = self._apply_dir((cx, cy), d)

        # HARD BAN on confirmed traps
        if (nx, ny) in self.confirmed_trapdoors:
            return -1e12

        safety = self._get_tile_safety(nx, ny)

        # === REVISIT HANDLING ===
        if is_revisit:
            base = -500 + safety * 0.5
            
            # If stuck in loop, add randomness and penalize recent tiles
            if self._is_stuck_in_loop():
                if (nx, ny) in self.recent_positions[-4:]:
                    base -= 200  # Avoid tiles we just visited
                base += np.random.random() * 100  # Add randomness to break loop
            
            return base

        util = 0.0

        # === EGG PRIORITY (high) ===
        if mt == MoveType.EGG:
            util += 400
            # Corner eggs are valuable
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 200

        # === SAFETY SCORING ===
        # Trust our safety calculations - but be VERY careful about danger!
        if safety < -100:
            # VERY dangerous - almost certainly a trap
            util -= 600
        elif safety < -30:
            # Dangerous - strong avoidance
            util -= 400
        elif safety < 0:
            # Somewhat risky
            util -= 150
        elif safety > 50:
            # Confirmed safe - go confidently!
            util += 80
        
        # Safety factor
        util += safety * 0.8

        # === EXPLORATION BONUS ===
        # Prefer tiles that open up more unexplored areas
        unexplored_neighbors = 0
        for test_d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            nnx, nny = self._apply_dir((nx, ny), test_d)
            if 0 <= nnx < self.board_size and 0 <= nny < self.board_size:
                if (nnx, nny) not in self.visited:
                    unexplored_neighbors += 1
        util += unexplored_neighbors * 15

        # === LIGHT DIRECTION MOMENTUM ===
        # Less than Pirate to avoid getting stuck on edges
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 20  # Reduced from 50
            elif d == self._opposite_dir(self.last_direction):
                util -= 40  # Reduced from 100

        # === TURD SCORING ===
        if mt == MoveType.TURD:
            if self.turn_index < 20:
                util -= 300  # Save turds
            else:
                # Basic turd scoring
                try:
                    ox, oy = board_state.chicken_enemy.get_location()
                    dist = abs(cx - ox) + abs(cy - oy)
                    if dist <= 4:
                        util += 100  # Nearby opponent
                    else:
                        util -= 100  # Too far
                except:
                    util -= 100

        return util + np.random.random() * 0.1

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

        # PRIORITY 1: EGG moves to NEW safe tiles
        egg_new = [m for m in new_moves if m[1] == MoveType.EGG]
        if egg_new:
            scored = [(self._score_move(m, board_state, cx, cy, False), m) for m in egg_new]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > -200:  # More aggressive threshold
                return scored[0][1]

        # PRIORITY 2: Any move to NEW tiles (if safe enough)
        if new_moves:
            scored = [(self._score_move(m, board_state, cx, cy, False), m) for m in new_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > -200:  # More aggressive threshold
                return scored[0][1]

        # PRIORITY 3: EGG moves even if revisit
        egg_revisit = [m for m in revisit_moves if m[1] == MoveType.EGG]
        if egg_revisit:
            scored = [(self._score_move(m, board_state, cx, cy, True), m) for m in egg_revisit]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # PRIORITY 4: Best revisit (with loop breaking)
        if revisit_moves:
            scored = [(self._score_move(m, board_state, cx, cy, True), m) for m in revisit_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        # PRIORITY 5: Only take risky new moves if not TOO dangerous
        if new_moves:
            scored = [(self._score_move(m, board_state, cx, cy, False), m) for m in new_moves]
            scored.sort(key=lambda x: x[0], reverse=True)
            # Only take if score > -400 (not extremely dangerous)
            if scored[0][0] > -400:
                return scored[0][1]
            # Otherwise fall through to any move

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
        
        # Track recent positions for loop detection
        self.recent_positions.append((cx, cy))
        if len(self.recent_positions) > 10:
            self.recent_positions.pop(0)
        
        # Update safety map from sensor data
        self._update_safety_from_sensors((cx, cy), sensors)

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

