"""
CornerHunter - Aggressive corner-first egg strategy

Philosophy:
- Corners give 3 eggs each = highest value
- Only avoid squares if we've FELT a trapdoor nearby
- Maximize eggs, minimize safety paranoia
"""

from typing import Optional, Tuple, Set

class PlayerAgent:
    """Corner-hunting chicken."""
    
    def __init__(self, board, time_left):
        self.my_parity: Optional[int] = None
        self.dangerous_squares: Set[Tuple[int, int]] = set()
        self.turn = 0
        
    def _apply_direction(self, pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Apply direction."""
        x, y = pos
        return [
            (x, y - 1),  # UP
            (x + 1, y),  # RIGHT
            (x, y + 1),  # DOWN
            (x - 1, y),  # LEFT
        ][direction]
    
    def _is_corner(self, x: int, y: int) -> bool:
        """Check if corner."""
        return (x, y) in [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    def play(self, board, sensors, time_left):
        """Play aggressively prioritizing corners."""
        self.turn += 1
        moves = board.get_valid_moves()
        
        if not moves:
            return None
        
        cur_x, cur_y = board.chicken_player.get_location()
        
        # Detect parity from FIRST egg destination
        if self.my_parity is None:
            for direction, move_type in moves:
                if move_type == 1:
                    dest = self._apply_direction((cur_x, cur_y), direction)
                    self.my_parity = (dest[0] + dest[1]) % 2
                    break
        
        # Only mark dangerous if we FELT (not just heard)
        heard, felt = sensors
        if felt and self.my_parity is not None:
            # Only mark immediate neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cur_x + dx, cur_y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        if (nx + ny) % 2 == self.my_parity:
                            self.dangerous_squares.add((nx, ny))
        
        # Score moves
        best_score = float('-inf')
        best_move = moves[0]
        
        for direction, move_type in moves:
            dest = self._apply_direction((cur_x, cur_y), direction)
            dest_x, dest_y = dest
            
            score = 0
            
            if move_type == 1:  # EGG
                score = 10000
                
                # HUGE corner bonus
                if self._is_corner(dest_x, dest_y):
                    score += 50000
                
                # Edge bonus
                if dest_x == 0 or dest_x == 7 or dest_y == 0 or dest_y == 7:
                    score += 5000
                
                # Penalize dangerous squares (but not too much)
                if dest in self.dangerous_squares:
                    score -= 8000
                
                # Slight center preference
                center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
                score -= center_dist * 100
                
            elif move_type == 0:  # PLAIN
                # Move toward nearest corner or center
                corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
                min_corner_dist = min(abs(dest_x - cx) + abs(dest_y - cy) for cx, cy in corners)
                score = -min_corner_dist * 50
                
                # Avoid dangerous
                if dest in self.dangerous_squares:
                    score -= 2000
                    
            elif move_type == 2:  # TURD
                # Only use turds if they block opponent significantly
                # For now, deprioritize
                score = -5000
            
            if score > best_score:
                best_score = score
                best_move = (direction, move_type)
        
        return best_move
