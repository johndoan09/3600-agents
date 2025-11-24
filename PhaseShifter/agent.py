"""
PhaseShifter - Phase-based strategy inspired by Bertha

Strategy:
- Early game (turns 0-13): Aggressive egg-laying, low trapdoor penalty, corner priority
- Mid game (turns 14-26): Balanced approach, moderate penalties, map exploration  
- Late game (turns 27-40): Conservative, high trapdoor penalty, safe eggs only
- Smooth transitions with dynamic parameter adjustment
"""

import numpy as np
from typing import Optional, Tuple, Set

class PlayerAgent:
    """Phase-based chicken agent with dynamic strategy."""
    
    def __init__(self, board, time_left):
        # Parity detection
        self.my_parity: Optional[int] = None
        self.opp_parity: Optional[int] = None
        
        # Bayesian trapdoor beliefs
        self.trapdoor_belief = np.ones((8, 8)) / 64.0
        
        # Game state
        self.turn = 0
        self.visited: Set[Tuple[int, int]] = set()
        
        # Phase thresholds
        self.early_phase_end = 13
        self.mid_phase_end = 26
        
    def _detect_parity(self, board, moves):
        """Detect parity from first egg move."""
        if self.my_parity is None:
            cur_x, cur_y = board.chicken_player.get_location()
            for direction, move_type in moves:
                if move_type == 1:
                    dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)
                    self.my_parity = (dest_x + dest_y) % 2
                    self.opp_parity = 1 - self.my_parity
                    break
    
    def _update_trapdoor_beliefs(self, board, sensors):
        """Bayesian update of trapdoor probabilities."""
        cur_x, cur_y = board.chicken_player.get_location()
        heard, felt = sensors
        
        if self.my_parity is None:
            return
        
        def get_sensor_prob(dx, dy):
            """Return (p_hear, p_feel) for offset."""
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
        """Check if position is corner."""
        return (x, y) in [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    def _is_edge(self, x: int, y: int) -> bool:
        """Check if position is edge (not corner)."""
        if self._is_corner(x, y):
            return False
        return x == 0 or x == 7 or y == 0 or y == 7
    
    def _get_phase_params(self):
        """Get dynamic parameters based on current phase."""
        if self.turn <= self.early_phase_end:
            # EARLY: VERY aggressive - maximize eggs quickly
            return {
                'trapdoor_penalty': 1500,  # Lower - take more risks
                'corner_bonus': 5000,      # Higher - corners are 3 eggs!
                'edge_bonus': 800,         # Higher
                'center_penalty': 3,       # Lower - don't avoid center
                'visit_penalty': 30        # Lower - revisit if needed
            }
        elif self.turn <= self.mid_phase_end:
            # MID: Balanced but still aggressive
            return {
                'trapdoor_penalty': 2800,  # Moderate
                'corner_bonus': 3500,      # Still prioritize corners
                'edge_bonus': 500,
                'center_penalty': 8,
                'visit_penalty': 80
            }
        else:
            # LATE: Conservative - only safe eggs
            return {
                'trapdoor_penalty': 7000,  # Very high - avoid risks
                'corner_bonus': 2000,      # Still valuable
                'edge_bonus': 300,
                'center_penalty': 12,
                'visit_penalty': 120
            }
    
    def _count_escape_routes(self, board, pos: Tuple[int, int]) -> int:
        """Count number of safe adjacent squares."""
        x, y = pos
        count = 0
        
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                # Check not blocked by turd
                if (nx, ny) not in board.turds_player and (nx, ny) not in board.turds_enemy:
                    # Check not adjacent to opponent turd
                    adjacent_to_opp_turd = False
                    for dx2, dy2 in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                        tx, ty = nx + dx2, ny + dy2
                        if (tx, ty) in board.turds_enemy:
                            adjacent_to_opp_turd = True
                            break
                    
                    if not adjacent_to_opp_turd:
                        count += 1
        
        return count
    
    def _score_move(self, board, direction, move_type) -> float:
        """Score move based on current phase."""
        params = self._get_phase_params()
        
        cur_pos = board.chicken_player.get_location()
        dest = self._apply_direction(cur_pos, direction)
        dest_x, dest_y = dest
        
        score = 0
        
        if move_type == 1:  # EGG
            score += 1000
            
            # Corner bonus
            if self._is_corner(dest_x, dest_y):
                score += params['corner_bonus']
            # Edge bonus
            elif self._is_edge(dest_x, dest_y):
                score += params['edge_bonus']
            
            # Trapdoor risk
            trap_prob = self.trapdoor_belief[dest_y, dest_x]
            score -= trap_prob * params['trapdoor_penalty']
            
            # Visit penalty (explore new squares)
            if dest in self.visited:
                score -= params['visit_penalty']
            
            # Center distance penalty
            center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
            score -= center_dist * params['center_penalty']
            
            # Mobility preservation
            escape_routes = self._count_escape_routes(board, dest)
            if escape_routes < 2:
                score -= 500  # Avoid getting trapped
            
        elif move_type == 0:  # PLAIN MOVE
            # Move toward unvisited squares
            if dest not in self.visited:
                score += 100
            
            # Avoid trapdoors
            trap_prob = self.trapdoor_belief[dest_y, dest_x]
            score -= trap_prob * (params['trapdoor_penalty'] / 2)
            
            # Move toward center
            center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
            score -= center_dist * 15
        
        return score
    
    def play(self, board, sensors, time_left):
        """Select best move based on phase."""
        self.turn += 1
        moves = board.get_valid_moves()
        
        if not moves:
            return None
        
        # Track position
        cur_pos = board.chicken_player.get_location()
        self.visited.add(cur_pos)
        
        # Detect parity
        self._detect_parity(board, moves)
        
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
