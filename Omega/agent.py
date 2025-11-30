from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


# ================================================================
# PlayerAgent – OMEGA (Fundamentally Different: Opponent-Centric)
# ================================================================
class PlayerAgent:
    """
    OMEGA: A fundamentally different approach based on lessons learned.
    
    Core Philosophy: "Follow the Leader, Block the Follower"
    
    Key Differences from Alpha/Zig:
    1. OPPONENT-CENTRIC: Prioritize tiles opponent has visited (PROVEN safe)
    2. ULTRA-SIMPLE SCORING: Fewer factors = fewer mistakes
    3. AGGRESSIVE SHADOWING: Follow opponent's safe path
    4. BINARY SAFETY: Tile is either SAFE (go) or DANGEROUS (avoid) - no gradients
    5. DETERMINISTIC: Minimal randomness for consistent behavior
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE
        self.my_egg_parity: Optional[int] = None

        # BINARY SAFETY MAP: 0 = unknown, 1 = confirmed safe, -1 = confirmed dangerous
        self.safety_status = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # Track danger signals (for probabilistic danger)
        self.danger_signals = np.zeros((self.board_size, self.board_size), dtype=int)

        # Confirmed traps
        self.confirmed_trapdoors: Set[Tuple[int, int]] = set()

        # VISITED TILES
        self.visited: Set[Tuple[int, int]] = set()

        # OPPONENT'S PATH - THE GOLD MINE (proven safe tiles!)
        self.opponent_path: Set[Tuple[int, int]] = set()
        
        # Our egg positions (for blocking)
        self.my_egg_positions: Set[Tuple[int, int]] = set()

        # Direction tracking
        self.last_direction: Optional[Direction] = None

        self.turn_index = 0
        
        # Track egg counts
        self.my_eggs = 0
        self.enemy_eggs = 0
        
        # Track opponent location
        self.enemy_loc: Optional[Tuple[int, int]] = None

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
        self.safety_status[y][x] = -1  # Confirmed dangerous

    def _process_sensors(self, loc, sensors):
        """
        SIMPLIFIED sensor processing:
        - Silence = ALL tiles within 2 are SAFE
        - Signal = tiles within range are SUSPICIOUS (accumulate danger)
        """
        (hw, fw), (hb, fb) = sensors
        lx, ly = loc
        
        # Mark current tile as definitely safe
        self.safety_status[ly][lx] = 1

        if hw or fw or hb or fb:
            # GOT A SIGNAL - mark nearby tiles as suspicious
            felt_signal = fw or fb
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if (nx, ny) in self.visited or (nx, ny) in self.opponent_path:
                            continue  # Already confirmed safe
                        
                        dist = abs(dx) + abs(dy)
                        if dist == 0:
                            continue
                        
                        if felt_signal and dist == 1:
                            # FELT adjacent = VERY dangerous
                            self.danger_signals[ny][nx] += 3
                        elif felt_signal and dist == 2:
                            self.danger_signals[ny][nx] += 1
                        elif dist == 1:
                            # Heard adjacent
                            self.danger_signals[ny][nx] += 2
                        elif dist == 2:
                            self.danger_signals[ny][nx] += 1
        else:
            # SILENCE = All tiles within 2 are SAFE!
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        dist = abs(dx) + abs(dy)
                        if dist <= 2:
                            self.safety_status[ny][nx] = 1  # Confirmed safe
                            self.danger_signals[ny][nx] = 0  # Clear any suspicion

    def _is_safe(self, x, y) -> bool:
        """Binary safety check"""
        if (x, y) in self.confirmed_trapdoors:
            return False
        if self.safety_status[y][x] == 1:
            return True
        if (x, y) in self.opponent_path:
            return True  # Opponent survived = safe
        if (x, y) in self.visited:
            return True  # We survived = safe
        return False
    
    def _is_dangerous(self, x, y) -> bool:
        """Check if tile has accumulated danger signals"""
        if (x, y) in self.confirmed_trapdoors:
            return True
        if self.safety_status[y][x] == -1:
            return True
        # High danger signal accumulation = dangerous
        if self.danger_signals[y][x] >= 4:
            return True
        return False
    
    def _distance_to_center(self, x, y) -> float:
        center = self.board_size / 2 - 0.5
        return abs(x - center) + abs(y - center)

    # =================================================================
    # BLOCKING STRATEGY (from Zig - it works!)
    # =================================================================
    def _score_egg_blocking(self, nx, ny) -> float:
        """Score egg placement for blocking potential"""
        score = 0.0
        
        # Line formation bonus
        for x_offset in [-2, -1, 1, 2]:
            if (nx + x_offset, ny) in self.my_egg_positions:
                score += 30
        for y_offset in [-2, -1, 1, 2]:
            if (nx, ny + y_offset) in self.my_egg_positions:
                score += 30
        
        # Edge eggs create barriers
        if nx == 0 or nx == self.board_size - 1 or ny == 0 or ny == self.board_size - 1:
            score += 40
        
        # Block opponent's path to corners
        if self.enemy_loc:
            ox, oy = self.enemy_loc
            corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
            for cx, cy in corners:
                opp_dist = abs(ox - cx) + abs(oy - cy)
                egg_dist = abs(nx - cx) + abs(ny - cy)
                if egg_dist < opp_dist and egg_dist <= 3:
                    score += 60
        
        return score

    def _score_move(self, move, board_state, cx, cy) -> float:
        """
        ULTRA-SIMPLIFIED SCORING
        
        Priority Order:
        1. NEVER go to confirmed dangerous tiles
        2. STRONGLY prefer opponent's path (proven safe)
        3. Prefer confirmed safe tiles
        4. Prefer new tiles over revisits
        5. Egg bonus + blocking
        6. Corner bonus
        """
        d, mt = move
        nx, ny = self._apply_dir((cx, cy), d)

        # ===== HARD BANS =====
        if (nx, ny) in self.confirmed_trapdoors:
            return -1e12
        
        if self._is_dangerous(nx, ny):
            return -1e9  # Very bad but not impossible

        util = 0.0
        
        is_new_tile = (nx, ny) not in self.visited
        is_opponent_path = (nx, ny) in self.opponent_path
        is_confirmed_safe = self._is_safe(nx, ny)
        
        # ===== SAFETY SCORING (SIMPLE!) =====
        if is_opponent_path and is_new_tile:
            # JACKPOT: New tile that opponent proved safe
            util += 400
        elif is_confirmed_safe and is_new_tile:
            # Good: New tile we know is safe
            util += 300
        elif is_opponent_path:
            # Opponent path but we visited = still good
            util += 150
        elif is_confirmed_safe:
            # Safe revisit
            util += 50
        elif is_new_tile:
            # Unknown new tile - risky but necessary
            danger = self.danger_signals[ny][nx]
            if danger == 0:
                util += 100  # No signals = probably fine
            elif danger <= 2:
                util -= 100  # Some suspicion
            else:
                util -= 300  # High suspicion
        else:
            # Revisiting unknown tile - worst option
            util -= 200

        # ===== EGG PRIORITY =====
        if mt == MoveType.EGG:
            util += 500
            # Corner eggs
            if (nx in (0, self.board_size - 1)) and (ny in (0, self.board_size - 1)):
                util += 250
            # Blocking bonus
            util += self._score_egg_blocking(nx, ny)

        # ===== DIRECTION MOMENTUM (mild) =====
        if self.last_direction is not None:
            if d == self.last_direction:
                util += 25
            elif d == self._opposite_dir(self.last_direction):
                util -= 40

        # ===== TURD SCORING =====
        if mt == MoveType.TURD:
            turds_left = board_state.chicken_player.get_turds_left()
            if turds_left <= 0 or self.turn_index < 20:
                util -= 500  # Too early
            else:
                # Place on opponent's parity if possible
                if self.my_egg_parity is not None:
                    opp_parity = 1 - self.my_egg_parity
                    if (cx + cy) % 2 == opp_parity:
                        util += 100
                    if self.enemy_loc:
                        dist = abs(cx - self.enemy_loc[0]) + abs(cy - self.enemy_loc[1])
                        if dist <= 4:
                            util += 80
                util -= 30  # Small penalty to not over-turd

        return util

    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        cx, cy = board_state.chicken_player.get_location()

        # Separate new vs revisit moves
        new_moves = []
        revisit_moves = []
        
        for m in moves:
            d, mt = m
            nx, ny = self._apply_dir((cx, cy), d)
            if (nx, ny) in self.visited:
                revisit_moves.append(m)
            else:
                new_moves.append(m)

        # Score all moves
        all_scored = [(self._score_move(m, board_state, cx, cy), m) for m in moves]
        all_scored.sort(key=lambda x: x[0], reverse=True)
        
        # PRIORITY: New tiles with acceptable scores
        if new_moves:
            new_scored = [(s, m) for s, m in all_scored if m in new_moves]
            if new_scored and new_scored[0][0] > -200:
                return new_scored[0][1]
        
        # EGG moves are always worth considering
        egg_moves = [(s, m) for s, m in all_scored if m[1] == MoveType.EGG]
        if egg_moves and egg_moves[0][0] > 0:
            return egg_moves[0][1]
        
        # Fall back to best overall
        if all_scored[0][0] > -1e8:
            return all_scored[0][1]
        
        return moves[0]

    def play(self, board_state: board.Board, sensors, time_left: Callable):
        self.turn_index += 1

        self.my_eggs = board_state.chicken_player.get_eggs_laid()
        self.enemy_eggs = board_state.chicken_enemy.get_eggs_laid()
        
        # ===== TRACK OPPONENT (THE KEY INSIGHT!) =====
        try:
            self.enemy_loc = board_state.chicken_enemy.get_location()
            if self.enemy_loc is not None:
                ex, ey = self.enemy_loc
                # OPPONENT SURVIVED THIS TILE = CONFIRMED SAFE
                if (ex, ey) not in self.opponent_path:
                    self.opponent_path.add((ex, ey))
                    self.safety_status[ey][ex] = 1
                    self.danger_signals[ey][ex] = 0
        except:
            self.enemy_loc = None

        # Record confirmed traps
        try:
            for loc in board_state.found_trapdoors:
                if loc not in self.confirmed_trapdoors:
                    self._record_trap(loc)
        except:
            pass

        cx, cy = board_state.chicken_player.get_location()
        
        # Process sensor data
        self._process_sensors((cx, cy), sensors)
        
        # Track our position
        self.visited.add((cx, cy))
        
        # Track egg placement
        moves = board_state.get_valid_moves()
        if moves:
            for m in moves:
                if m[1] == MoveType.EGG:
                    self.my_egg_positions.add((cx, cy))
                    break

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

