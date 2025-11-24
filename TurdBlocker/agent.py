"""
TurdBlocker - Strategic turd placement to block opponent

Strategy:
- Calculate opponent parity from their first egg
- Place turds ONLY when they block >=2 opponent egg squares
- Protect high-value squares (corners, edges)
- Otherwise always lay eggs
- Bayesian trapdoor tracking for safety
"""

import numpy as np
from typing import Optional, Tuple, Set

class PlayerAgent:
    """Agent focused on strategic turd placement."""
    
    def __init__(self, board, time_left):
        # Parity detection
        self.my_parity: Optional[int] = None
        self.opp_parity: Optional[int] = None
        
        # Bayesian trapdoor beliefs
        self.trapdoor_belief = np.ones((8, 8)) / 64.0
        
        # Game state
        self.turn = 0
        self.opp_egg_squares: Set[Tuple[int, int]] = set()
        self.my_egg_squares: Set[Tuple[int, int]] = set()
        
    def _detect_parity(self, board, moves):
        """Detect my parity from first egg move."""
        if self.my_parity is None:
            cur_x, cur_y = board.chicken_player.get_location()
            for direction, move_type in moves:
                if move_type == 1:
                    dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)
                    self.my_parity = (dest_x + dest_y) % 2
                    self.opp_parity = 1 - self.my_parity
                    break
    
    def _track_opponent_eggs(self, board):
        """Track opponent egg locations to infer parity."""
        for y in range(8):
            for x in range(8):
                if (x, y) in board.eggs_enemy:
                    self.opp_egg_squares.add((x, y))
                    if self.opp_parity is None:
                        self.opp_parity = (x + y) % 2
                        self.my_parity = 1 - self.opp_parity
    
    def _update_trapdoor_beliefs(self, board, sensors):
        """Bayesian update of trapdoor probabilities."""
        cur_x, cur_y = board.chicken_player.get_location()
        heard, felt = sensors
        
        if self.my_parity is None:
            return
        
        def get_sensor_prob(dx, dy):
            dist = abs(dx) + abs(dy)
            if dist == 1:
                return 0.5, 0.3
            elif dx != 0 and dy != 0 and abs(dx) == 1 and abs(dy) == 1:
                return 0.25, 0.15
            elif dist == 2:
                return 0.1, 0.0
            return 0.0, 0.0
        
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 != self.my_parity:
                    continue
                
                dx, dy = x - cur_x, y - cur_y
                p_hear, p_feel = get_sensor_prob(dx, dy)
                
                prior = self.trapdoor_belief[y, x]
                
                if heard and felt:
                    likelihood = p_hear * p_feel
                elif heard:
                    likelihood = p_hear * (1 - p_feel)
                elif felt:
                    likelihood = (1 - p_hear) * p_feel
                else:
                    likelihood = (1 - p_hear) * (1 - p_feel)
                
                self.trapdoor_belief[y, x] = likelihood * prior
        
        total = np.sum(self.trapdoor_belief)
        if total > 0:
            self.trapdoor_belief /= total
    
    def _apply_direction(self, pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Apply direction to position."""
        x, y = pos
        if direction == 0:
            return x, y - 1
        elif direction == 1:
            return x + 1, y
        elif direction == 2:
            return x, y + 1
        else:
            return x - 1, y
    
    def _is_corner(self, x: int, y: int) -> bool:
        """Check if corner."""
        return (x, y) in [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    def _is_edge(self, x: int, y: int) -> bool:
        """Check if edge (not corner)."""
        if self._is_corner(x, y):
            return False
        return x == 0 or x == 7 or y == 0 or y == 7
    
    def _count_blocked_opp_eggs(self, board, turd_pos: Tuple[int, int]) -> int:
        """Count how many opponent egg squares would be blocked by turd at pos."""
        if self.opp_parity is None:
            return 0
        
        tx, ty = turd_pos
        blocked = 0
        
        # Check all adjacent squares
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                # Is it opponent parity?
                if (nx + ny) % 2 == self.opp_parity:
                    # Is it unoccupied?
                    if (nx, ny) not in board.eggs_enemy and (nx, ny) not in board.turds_enemy:
                        if (nx, ny) not in board.eggs_player and (nx, ny) not in board.turds_player:
                            blocked += 1
        
        return blocked
    
    def _should_place_turd(self, board, pos: Tuple[int, int]) -> bool:
        """Decide if we should place turd at this position."""
        # Count how many opponent eggs we'd block
        blocked = self._count_blocked_opp_eggs(board, pos)
        
        # Place turd if blocking >=2 opponent eggs
        if blocked >= 2:
            return True
        
        # Place turd if protecting a corner we haven't used yet
        if self._is_corner(pos[0], pos[1]):
            if pos not in self.my_egg_squares:
                # Only if opponent is nearby (within 3 squares)
                opp_x, opp_y = board.chicken_enemy.get_location()
                dist = abs(pos[0] - opp_x) + abs(pos[1] - opp_y)
                if dist <= 3:
                    return True
        
        return False
    
    def _score_move(self, board, direction, move_type) -> float:
        """Score move."""
        cur_pos = board.chicken_player.get_location()
        dest = self._apply_direction(cur_pos, direction)
        dest_x, dest_y = dest
        
        score = 0
        
        if move_type == 1:  # EGG
            score += 1000
            
            # Corner bonus (3 eggs total)
            if self._is_corner(dest_x, dest_y):
                score += 4000
            # Edge bonus
            elif self._is_edge(dest_x, dest_y):
                score += 1000
            
            # Trapdoor risk
            trap_prob = self.trapdoor_belief[dest_y, dest_x]
            score -= trap_prob * 4000
            
            # Prefer unvisited
            if dest in self.my_egg_squares:
                score -= 200
            
        elif move_type == 2:  # TURD
            # Check if strategic
            if self._should_place_turd(board, dest):
                blocked = self._count_blocked_opp_eggs(board, dest)
                score += 2000 + (blocked * 500)  # Reward blocking
            else:
                score -= 10000  # Heavily penalize non-strategic turds
            
        elif move_type == 0:  # PLAIN MOVE
            # Move toward center
            center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
            score -= center_dist * 20
            
            # Avoid trapdoors
            trap_prob = self.trapdoor_belief[dest_y, dest_x]
            score -= trap_prob * 1000
        
        return score
    
    def play(self, board, sensors, time_left):
        """Select best move."""
        self.turn += 1
        moves = board.get_valid_moves()
        
        if not moves:
            return None
        
        # Track my eggs
        for y in range(8):
            for x in range(8):
                if (x, y) in board.eggs_player:
                    self.my_egg_squares.add((x, y))
        
        # Detect parity
        self._detect_parity(board, moves)
        self._track_opponent_eggs(board)
        
        # Update beliefs
        self._update_trapdoor_beliefs(board, sensors)
        
        # Score and select best move
        best_score = float('-inf')
        best_move = moves[0]
        
        for direction, move_type in moves:
            score = self._score_move(board, direction, move_type)
            if score > best_score:
                best_score = score
                best_move = (direction, move_type)
        
        return best_move
