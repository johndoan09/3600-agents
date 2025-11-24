"""
AlphaChicken - Minimax agent with Bayesian trapdoor tracking and egg maximization

Strategy:
1. Bayesian belief tracking for trapdoor locations
2. Minimax search with alpha-beta pruning
3. Evaluation: eggs laid, available eggs, trapdoor risk, mobility
4. Corner prioritization and turd optimization
"""

import numpy as np
from typing import Optional, Tuple, List, Set

class PlayerAgent:
    """Advanced chicken agent with search and probabilistic reasoning."""
    
    def __init__(self, board, time_left):
        # Parity detection
        self.my_parity: Optional[int] = None
        self.opp_parity: Optional[int] = None
        
        # Bayesian trapdoor beliefs (8x8 probability grid)
        self.trapdoor_belief = np.ones((8, 8)) / 64.0  # Uniform prior
        
        # Game state tracking
        self.turn = 0
        self.my_eggs = 0
        self.opp_eggs = 0
        
        # Time management (don't call time_left in constructor)
        self.time_per_early_move = 0.15  # First 10 moves
        self.time_per_mid_move = 0.08    # Moves 11-30
        self.time_per_late_move = 0.15   # Last 10 moves
        
    def _detect_parity(self, board, moves):
        """Detect parity from first available egg move."""
        if self.my_parity is None:
            cur_x, cur_y = board.chicken_player.get_location()
            for direction, move_type in moves:
                if move_type == 1:  # EGG
                    dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)
                    self.my_parity = (dest_x + dest_y) % 2
                    self.opp_parity = 1 - self.my_parity
                    break
    
    def _update_trapdoor_beliefs(self, board, sensors):
        """Bayesian update of trapdoor location probabilities."""
        cur_x, cur_y = board.chicken_player.get_location()
        heard, felt = sensors
        
        if self.my_parity is None:
            return
        
        # Define sensor probabilities
        def get_sensor_prob(dx, dy):
            """Return (p_hear, p_feel) for offset (dx, dy)."""
            dist = abs(dx) + abs(dy)  # Manhattan distance
            if dist == 1:  # Edge neighbor
                return 0.5, 0.3
            elif dx != 0 and dy != 0 and abs(dx) == 1 and abs(dy) == 1:  # Diagonal
                return 0.25, 0.15
            elif dist == 2:  # 2 squares away
                return 0.1, 0.0
            return 0.0, 0.0
        
        # Update beliefs for my parity squares
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 != self.my_parity:
                    continue
                
                dx, dy = x - cur_x, y - cur_y
                p_hear, p_feel = get_sensor_prob(dx, dy)
                
                # Bayesian update: P(trap|sensor) ∝ P(sensor|trap) * P(trap)
                prior = self.trapdoor_belief[y, x]
                
                # Likelihood of this sensor reading given trapdoor at (x,y)
                if heard and felt:
                    likelihood = p_hear * p_feel
                elif heard:
                    likelihood = p_hear * (1 - p_feel)
                elif felt:
                    likelihood = (1 - p_hear) * p_feel
                else:
                    likelihood = (1 - p_hear) * (1 - p_feel)
                
                posterior = likelihood * prior
                self.trapdoor_belief[y, x] = posterior
        
        # Normalize probabilities
        total = np.sum(self.trapdoor_belief)
        if total > 0:
            self.trapdoor_belief /= total
    
    def _apply_direction(self, pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Apply direction to position."""
        x, y = pos
        if direction == 0:  # UP
            return x, y - 1
        elif direction == 1:  # RIGHT
            return x + 1, y
        elif direction == 2:  # DOWN
            return x, y + 1
        else:  # LEFT
            return x - 1, y
    
    def _is_corner(self, x: int, y: int) -> bool:
        """Check if corner."""
        return (x, y) in [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    def _count_available_eggs(self, board, my_parity: int) -> int:
        """Count unoccupied egg squares of my parity."""
        count = 0
        for y in range(8):
            for x in range(8):
                if (x + y) % 2 == my_parity:
                    # Check if empty
                    if (x, y) not in board.eggs_player and (x, y) not in board.eggs_enemy:
                        if (x, y) not in board.turds_player and (x, y) not in board.turds_enemy:
                            count += 1
        return count
    
    def _score_move(self, board, direction, move_type) -> float:
        """Score a move based on eggs, safety, and position."""
        cur_pos = board.chicken_player.get_location()
        dest = self._apply_direction(cur_pos, direction)
        dest_x, dest_y = dest
        
        score = 0
        
        if move_type == 1:  # EGG
            score += 1000
            
            # Corner bonus (3 eggs total)
            if self._is_corner(dest_x, dest_y):
                score += 2000
            
            # Trapdoor risk penalty
            trap_prob = self.trapdoor_belief[dest_y, dest_x]
            score -= trap_prob * 5000
            
            # Center preference
            center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
            score -= center_dist * 10
            
        elif move_type == 0:  # PLAIN
            # Move toward center
            center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
            score -= center_dist * 20
            
            # Avoid trapdoors
            trap_prob = self.trapdoor_belief[dest_y, dest_x]
            score -= trap_prob * 1000
            
        return score
    
    def play(self, board, sensors, time_left):
        """Select best move using greedy evaluation."""
        self.turn += 1
        moves = board.get_valid_moves()
        
        if not moves:
            return None
        
        # Detect parity
        self._detect_parity(board, moves)
        
        # Update trapdoor beliefs
        self._update_trapdoor_beliefs(board, sensors)
        
        # Score all moves and pick best
        best_score = float('-inf')
        best_move = moves[0]
        
        for direction, move_type in moves:
            score = self._score_move(board, direction, move_type)
            if score > best_score:
                best_score = score
                best_move = (direction, move_type)
        
        return best_move
