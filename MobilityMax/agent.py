"""
MobilityMax - Egg maximization while preserving mobility

Strategy:
- Always ensure >=2 escape routes after each move
- Lay eggs whenever safe and mobile
- Use turds to block opponent, not ourselves
- Avoid trapdoors conservatively (only when felt)
"""

from typing import Optional, Tuple, Set

class PlayerAgent:
    """Mobility-preserving egg maximizer."""
    
    def __init__(self, board, time_left):
        self.my_parity: Optional[int] = None
        self.risky_squares: Set[Tuple[int, int]] = set()
        self.turn = 0
        self.visited_eggs: Set[Tuple[int, int]] = set()
        
    def _apply_direction(self, pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Apply direction."""
        x, y = pos
        return [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)][direction]
    
    def _is_valid_square(self, pos: Tuple[int, int], board) -> bool:
        """Check if square is walkable."""
        x, y = pos
        if not (0 <= x < 8 and 0 <= y < 8):
            return False
        
        # Can't walk on opponent's eggs or turds
        if pos in board.eggs_enemy or pos in board.turds_enemy:
            return False
        
        # Can't walk adjacent to opponent's turds
        opp_x, opp_y = board.chicken_enemy.get_location()
        for tx, ty in board.turds_enemy:
            if abs(x - tx) + abs(y - ty) == 1:  # Adjacent
                return False
        
        return True
    
    def _count_escape_routes(self, pos: Tuple[int, int], board) -> int:
        """Count valid escape routes from position."""
        count = 0
        for direction in range(4):
            next_pos = self._apply_direction(pos, direction)
            if self._is_valid_square(next_pos, board):
                count += 1
        return count
    
    def _is_corner(self, x: int, y: int) -> bool:
        """Check if corner."""
        return (x, y) in [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    def play(self, board, sensors, time_left):
        """Play with mobility preservation."""
        self.turn += 1
        moves = board.get_valid_moves()
        
        if not moves:
            return None
        
        cur_pos = board.chicken_player.get_location()
        cur_x, cur_y = cur_pos
        
        # Detect parity
        if self.my_parity is None:
            for direction, move_type in moves:
                if move_type == 1:
                    dest = self._apply_direction(cur_pos, direction)
                    self.my_parity = (dest[0] + dest[1]) % 2
                    break
        
        # Track risky squares (only when felt)
        heard, felt = sensors
        if felt and self.my_parity is not None:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cur_x + dx, cur_y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        if (nx + ny) % 2 == self.my_parity:
                            self.risky_squares.add((nx, ny))
        
        # Filter moves that preserve mobility
        safe_moves = []
        for direction, move_type in moves:
            dest = self._apply_direction(cur_pos, direction)
            
            # Skip if risky trapdoor square
            if dest in self.risky_squares:
                continue
            
            # Check mobility after this move
            escape_routes = self._count_escape_routes(dest, board)
            
            # Require at least 2 escape routes (or accept if it's our only option)
            if escape_routes >= 2 or len(moves) <= 3:
                safe_moves.append((direction, move_type, escape_routes))
        
        # If no safe moves, take any move
        if not safe_moves:
            safe_moves = [(d, t, 1) for d, t in moves]
        
        # Score moves
        best_score = float('-inf')
        best_move = safe_moves[0][:2]
        
        for direction, move_type, escape_routes in safe_moves:
            dest = self._apply_direction(cur_pos, direction)
            dest_x, dest_y = dest
            
            score = 0
            
            if move_type == 1:  # EGG
                score = 10000
                
                # Corner bonus
                if self._is_corner(dest_x, dest_y):
                    score += 5000
                
                # Mobility bonus
                score += escape_routes * 500
                
                # Prefer unvisited egg squares
                if dest not in self.visited_eggs:
                    score += 2000
                    
                # Center slight preference
                center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
                score -= center_dist * 50
                
            elif move_type == 0:  # PLAIN
                # Mobility is key
                score = escape_routes * 300
                
                # Move toward unvisited egg parity squares
                if self.my_parity is not None and (dest_x + dest_y) % 2 == self.my_parity:
                    if dest not in self.visited_eggs and dest not in board.eggs_player:
                        score += 1000
                
            elif move_type == 2:  # TURD
                # Use turds strategically - save for blocking opponent
                opp_pos = board.chicken_enemy.get_location()
                dist_to_opp = abs(dest_x - opp_pos[0]) + abs(dest_y - opp_pos[1])
                
                if dist_to_opp <= 3:  # Close to opponent
                    score = 500
                else:
                    score = -1000  # Don't waste turds
            
            if score > best_score:
                best_score = score
                best_move = (direction, move_type)
        
        # Track visited eggs
        if best_move[1] == 1:
            dest = self._apply_direction(cur_pos, best_move[0])
            self.visited_eggs.add(dest)
        
        return best_move
