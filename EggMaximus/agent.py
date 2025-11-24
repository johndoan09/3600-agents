"""
EggMaximus - Pure egg maximization strategy

Focus: Lay as many eggs as possible while avoiding trapdoors
"""

from typing import Optional, Tuple

class PlayerAgent:
    """Greedy egg-laying agent."""
    
    def __init__(self, board, time_left):
        self.my_parity: Optional[int] = None
        self.turn = 0
        self.risky_squares = set()
        
    def _is_corner(self, x: int, y: int) -> bool:
        """Check if position is a corner."""
        return (x, y) in [(0, 0), (0, 7), (7, 0), (7, 7)]
    
    def _apply_direction(self, pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Apply direction to position."""
        x, y = pos
        if direction == 0:  # UP
            return x, y - 1
        elif direction == 1:  # RIGHT
            return x + 1, y
        elif direction == 2:  # DOWN
            return x, y + 1
        else:  # LEFT (3)
            return x - 1, y
    
    def play(self, board, sensors, time_left):
        """Select move prioritizing eggs."""
        self.turn += 1
        
        moves = board.get_valid_moves()
        if not moves:
            return None
        
        cur_x, cur_y = board.chicken_player.get_location()
        
        # Detect parity from FIRST EGG DESTINATION (not current position!)
        if self.my_parity is None:
            for direction, move_type in moves:
                if move_type == 1:  # EGG
                    dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)
                    self.my_parity = (dest_x + dest_y) % 2
                    break
        
        # Track trapdoor sensors
        heard = sensors[0] if len(sensors) > 0 else False
        felt = sensors[1] if len(sensors) > 1 else False
        
        if (heard or felt) and self.my_parity is not None:
            # Mark nearby same-parity squares as risky
            radius = 2 if heard else 1
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cur_x + dx, cur_y + dy
                    if 0 <= nx < 8 and 0 <= ny < 8:
                        if (nx + ny) % 2 == self.my_parity:
                            self.risky_squares.add((nx, ny))
        
        # Prioritize moves
        egg_moves = []
        plain_moves = []
        
        for direction, move_type in moves:
            dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)
            
            if move_type == 1:  # EGG
                priority = 1000
                
                # Corners give 3 eggs total
                if self._is_corner(dest_x, dest_y):
                    priority += 2000
                
                # Correct parity
                if self.my_parity is not None and (dest_x + dest_y) % 2 == self.my_parity:
                    priority += 100
                
                # AVOID risky squares
                if (dest_x, dest_y) in self.risky_squares:
                    priority -= 5000
                
                # Prefer center
                center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
                priority -= center_dist * 10
                
                egg_moves.append((priority, direction, move_type))
            
            elif move_type == 0:  # PLAIN
                priority = 0
                
                # Move toward center
                center_dist = abs(dest_x - 3.5) + abs(dest_y - 3.5)
                priority -= center_dist * 10
                
                # Avoid risky squares
                if (dest_x, dest_y) in self.risky_squares:
                    priority -= 1000
                
                plain_moves.append((priority, direction, move_type))
        
        # Choose best egg move, otherwise best plain move
        if egg_moves:
            egg_moves.sort(reverse=True)
            return (egg_moves[0][1], egg_moves[0][2])
        elif plain_moves:
            plain_moves.sort(reverse=True)
            return (plain_moves[0][1], plain_moves[0][2])
        else:
            return moves[0]
