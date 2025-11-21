from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    BALANCED TACTICAL AGENT
    ----------------------------------------
    - Mix of offense + safety
    - Applies pressure to opponent without overcommitting
    - Uses turds strategically (offensive when close, defensive when cornered)
    - Moderate trapdoor fear
    - Good edge rotation, avoids camping
    - Avoids loops and dead travel
    - Shallow minimax for tactical play
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.board_size = board.game_map.MAP_SIZE

        # Movement tracking
        self.start_x, self.start_y = board.chicken_player.get_location()
        self.prev_loc = None
        self.visited_counts: Dict[Tuple[int, int], int] = {}
        self.recent_positions: List[Tuple[int, int]] = []
        self.max_recent_positions = 6

        # Egg & edge memory
        self.egg_squares = set()
        self.my_egg_parity: Optional[int] = None
        self.edge_side_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.last_edge_side = None

        # Trapdoor beliefs
        self.belief_white = np.zeros((self.board_size, self.board_size))
        self.belief_black = np.zeros((self.board_size, self.board_size))
        self._init_trapdoor_beliefs()

        # Balanced fear
        self.trapdoor_penalty = 70.0

        self.turn_index = 0
        self.gamma = 0.45  # slightly deeper lookahead value

    # -------------------------------------------------------
    # Trapdoor belief
    # -------------------------------------------------------
    def _init_trapdoor_beliefs(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x + y) % 2 == 0:
                    self.belief_white[y][x] = 1
                else:
                    self.belief_black[y][x] = 1
        self.belief_white /= self.belief_white.sum()
        self.belief_black /= self.belief_black.sum()

    def _trapdoor_risk_at(self, x, y):
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return (self.belief_white[y][x] + self.belief_black[y][x])
        return 0

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------
    def _enum(self, obj):
        return obj.name if hasattr(obj, "name") else str(obj)

    def _unpack_move(self, move):
        return move[0], move[1]

    def _apply_direction(self, loc, direction):
        x, y = loc
        d = self._enum(direction).lower()
        if "up" in d:
            return x, y - 1
        if "down" in d:
            return x, y + 1
        if "left" in d:
            return x - 1, y
        if "right" in d:
            return x + 1, y
        return x, y

    def _get_opponent_location(self, board_state):
        return board_state.chicken_enemy.get_location()

    def _edge_side(self, x, y):
        if y == 0:
            return 0
        if y == self.board_size - 1:
            return 1
        if x == 0:
            return 2
        if x == self.board_size - 1:
            return 3
        return None

    # -------------------------------------------------------
    # Spatial penalties
    # -------------------------------------------------------
    def _visited_penalty(self, x, y):
        c = self.visited_counts.get((x, y), 0)
        return min(5, c - 1) if c > 1 else 0

    def _recent_penalty(self, x, y):
        p = 0
        for (px, py) in self.recent_positions:
            d = abs(px - x) + abs(py - y)
            if d == 0:
                p += 2
            elif d == 1:
                p += 1
        return p

    def _backtrack_penalty(self, dest):
        if self.prev_loc == dest:
            return 7.0
        return 0

    # -------------------------------------------------------
    # CORE UTILITY FUNCTION
    # -------------------------------------------------------
    def _immediate_utility_of_move(self, move, board_state, sensors):
        direction, move_type = self._unpack_move(move)
        name = self._enum(move_type).lower()

        cur_x, cur_y = board_state.chicken_player.get_location()
        dest_x, dest_y = self._apply_direction((cur_x, cur_y), direction)
        dest = (dest_x, dest_y)

        util = 0

        # ----------------------------------------------------
        # Balanced egg logic
        # ----------------------------------------------------
        if "egg" in name:
            util += 100
            if self.my_egg_parity and (dest_x + dest_y) % 2 == self.my_egg_parity:
                util += 4
        else:
            util += 1

        # ----------------------------------------------------
        # TURD LOGIC (BALANCED)
        # ----------------------------------------------------
        if "turd" in name:
            ox, oy = self._get_opponent_location(board_state)
            dist = abs(ox - dest_x) + abs(oy - dest_y)

            # Offensive turds at close range
            if dist <= 2:
                util += 20
            elif dist <= 3:
                util += 8
            else:
                util -= 8

            # Defensive turf: block opponent approach
            if dist == 1:
                util += 10

            # Check if this turd blocks most opponent moves
            opp_moves = board_state.get_valid_moves(enemy=True)
            blocks = True
            for om in opp_moves:
                od, ot = self._unpack_move(om)
                ox2, oy2 = self._apply_direction((ox, oy), od)
                if (ox2, oy2) != (dest_x, dest_y):
                    blocks = False
                    break
            if blocks:
                util += 40

        # ----------------------------------------------------
        # MODERATE PURSUIT (less than style A)
        # ----------------------------------------------------
        ox, oy = self._get_opponent_location(board_state)
        before = abs(cur_x - ox) + abs(cur_y - oy)
        after = abs(dest_x - ox) + abs(dest_y - oy)

        # Chase if safe and not entering a trap zone
        if after < before:
            util += 2
        else:
            util -= 0.5

        # ----------------------------------------------------
        # Trapdoor penalty
        # ----------------------------------------------------
        util -= self.trapdoor_penalty * self._trapdoor_risk_at(dest_x, dest_y)

        # ----------------------------------------------------
        # Edge rotation logic (balanced)
        # ----------------------------------------------------
        side = self._edge_side(dest_x, dest_y)
        if side is not None:
            util += 1.4
            if self.edge_side_counts[side] == 0:
                util += 3  # encourage reaching new edges
            if self.last_edge_side == side and self.edge_side_counts[side] > 7:
                util -= 1.3  # discourage camping

        # ----------------------------------------------------
        # Anti-loop
        # ----------------------------------------------------
        util -= self._visited_penalty(dest_x, dest_y)
        util -= self._recent_penalty(dest_x, dest_y)
        util -= self._backtrack_penalty(dest)

        if dest in self.egg_squares:
            util -= 3

        return util + np.random.random() * 0.01

    # -------------------------------------------------------
    # Evaluation for minimax
    # -------------------------------------------------------
    def _evaluate_board(self, board_state):
        x, y = board_state.chicken_player.get_location()
        v = 0
        v -= self.trapdoor_penalty * self._trapdoor_risk_at(x, y)
        v -= self._visited_penalty(x, y)
        return v

    # -------------------------------------------------------
    # Alpha-beta minimax
    # -------------------------------------------------------
    def _alpha_beta(self, board_state, depth, alpha, beta, maximizing, time_left, sensors):
        if board_state is None:
            return -1e9

        try:
            if time_left() < 0.3:
                return self._evaluate_board(board_state)
        except:
            pass

        moves = board_state.get_valid_moves()
        if depth == 0 or not moves:
            return self._evaluate_board(board_state)

        if maximizing:
            val = -1e9
            for m in moves:
                d, t = self._unpack_move(m)
                nb = board_state.forecast_move(d, t)
                if nb:
                    try:
                        nb = nb.reverse_perspective()
                    except:
                        pass
                    v = self._alpha_beta(
                        nb, depth - 1, alpha, beta, False, time_left, sensors)
                    val = max(val, v)
                    alpha = max(alpha, val)
                    if alpha >= beta:
                        break
            return val
        else:
            val = 1e9
            for m in moves:
                d, t = self._unpack_move(m)
                nb = board_state.forecast_move(d, t)
                if nb:
                    try:
                        nb = nb.reverse_perspective()
                    except:
                        pass
                    v = self._alpha_beta(
                        nb, depth - 1, alpha, beta, True, time_left, sensors)
                    val = min(val, v)
                    beta = min(beta, val)
                    if alpha >= beta:
                        break
            return val

    # -------------------------------------------------------
    # Move chooser
    # -------------------------------------------------------
    def _choose_move(self, board_state, sensors, time_left):
        moves = board_state.get_valid_moves()
        if not moves:
            return None

        # Egg moves first
        egg_moves = [m for m in moves if "egg" in self._enum(m[1]).lower()]
        if egg_moves:
            return max(egg_moves, key=lambda m: self._immediate_utility_of_move(m, board_state, sensors))

        # Score moves
        scored = [(self._immediate_utility_of_move(
            m, board_state, sensors), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Minimax depth based on remaining time
        try:
            time_left_val = time_left()
        except:
            time_left_val = 999

        depth = 3 if time_left_val > 150 else 2
        best_move = scored[0][1]
        best_val = -1e9

        for _, m in scored:
            d, t = self._unpack_move(m)
            immediate = self._immediate_utility_of_move(
                m, board_state, sensors)

            nb = board_state.forecast_move(d, t)
            if nb is None:
                total = immediate
            else:
                try:
                    nb = nb.reverse_perspective()
                except:
                    pass
                future = self._alpha_beta(
                    nb, depth - 1, -1e9, 1e9, False, time_left, sensors)
                total = immediate + self.gamma * future

            if total > best_val:
                best_val = total
                best_move = m

        return best_move

    # -------------------------------------------------------
    # Main play
    # -------------------------------------------------------
    def play(self, board_state, sensors, time_left):
        self.turn_index += 1

        moves = board_state.get_valid_moves()
        if not moves:
            return None

        loc = board_state.chicken_player.get_location()

        # Determine parity once
        if self.my_egg_parity is None:
            for m in moves:
                _, mt = self._unpack_move(m)
                if "egg" in self._enum(mt).lower():
                    x, y = loc
                    self.my_egg_parity = (x + y) % 2

        chosen = self._choose_move(board_state, sensors, time_left)
        if chosen is None:
            return moves[0]

        # Update tracking
        dir, move_type = self._unpack_move(chosen)
        dest_x, dest_y = self._apply_direction(loc, dir)
        dest = (dest_x, dest_y)

        self.prev_loc = loc
        self.visited_counts[dest] = self.visited_counts.get(dest, 0) + 1

        self.recent_positions.append(dest)
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)

        if "egg" in self._enum(move_type).lower():
            self.egg_squares.add(dest)

        side = self._edge_side(dest_x, dest_y)
        if side is not None:
            self.edge_side_counts[side] += 1
            self.last_edge_side = side

        return chosen
