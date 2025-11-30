from collections.abc import Callable
from typing import List, Set, Tuple, Optional

import numpy as np
from game import *
from game.enums import Direction, MoveType


class PlayerAgent:
    """
    Chad v7 – Territory racer + smart walls + Bayesian traps.

    - Bayesian beliefs for both trapdoors (white/black) updated from sensors.
    - Eggs-first policy: if a reasonably safe EGG move exists, always prefer it.
    - Strong early-game drive to cross to the far side of the board and fan out.
    - Frontier exploration bonus (dest with many unvisited neighbors).
    - Turds:
        * Rewarded only when near the enemy and wall-like (same row/col,
          enemy egg color, central).
        * Penalized when far from the enemy (useless turds).
        * Slight extra push to use remaining turds late game.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, board: "board.Board", time_left: Callable):
        self.board_size: int = board.game_map.MAP_SIZE

        # Spawn / color info
        self.spawn: Tuple[int, int] = board.chicken_player.get_location()
        self.my_parity: int = (self.spawn[0] + self.spawn[1]) % 2
        self.enemy_parity: int = 1 - self.my_parity

        # Path memory / loops
        self.prev_location: Optional[Tuple[int, int]] = None
        self.expected_next_location: Optional[Tuple[int, int]] = None
        self.visited: Set[Tuple[int, int]] = {self.spawn}
        self.recent_positions: List[Tuple[int, int]] = []

        # Bayesian trapdoor beliefs
        self.belief_white = np.zeros((self.board_size, self.board_size), dtype=float)
        self.belief_black = np.zeros((self.board_size, self.board_size), dtype=float)
        self._init_priors()

        # --- Strategy parameters --------------------------------------
        # Eggs & exploration
        self.egg_reward: float = 8.0
        self.explore_bonus: float = 2.5
        self.frontier_bonus_weight: float = 0.7  # per unvisited neighbor

        # Board geometry
        self.center_weight: float = 0.10
        self.backtrack_penalty: float = 2.0
        self.loop_penalty: float = 2.0

        # Trap avoidance (less paranoid so we cross mid-board)
        self.base_trap_penalty: float = 45.0
        self.base_near_trap_penalty: float = 5.0

        # Target / territory
        self.target_weight: float = 0.5
        self.cross_weight_early: float = 1.5
        self.cross_weight_mid: float = 0.6
        self.cross_early_turn: int = 12
        self.cross_mid_turn: int = 20

        # Turds / walls
        self.turd_block_weight: float = 4.0
        self.enemy_parity_block_bonus: float = 3.0
        self.useless_turd_penalty: float = 2.0
        self.early_turd_cutoff_turn: int = 16  # can still turd earlier if very good
        self.late_game_turn: int = 28  # after this, encourage spending turds

        # Pre-compute a "far side" crossing target
        self.cross_target: Tuple[int, int] = self._compute_cross_target()

    # ------------------------------------------------------------------
    # Belief helpers
    # ------------------------------------------------------------------
    def _init_priors(self) -> None:
        """
        Initialize priors: traps more likely in inner rings.
        Ring index = min(x, y, n-1-x, n-1-y).
        Ring 0/1 weight=0, ring 2 weight=1, ring 3 weight=2 (for 8x8).
        """
        n = self.board_size

        total_w = 0.0
        total_b = 0.0
        for x in range(n):
            for y in range(n):
                ring = min(x, y, n - 1 - x, n - 1 - y)
                weight = max(0, ring - 1)  # 0,0,1,2 ...

                if weight <= 0:
                    continue

                if (x + y) % 2 == 0:
                    self.belief_white[x, y] = weight
                    total_w += weight
                else:
                    self.belief_black[x, y] = weight
                    total_b += weight

        if total_w > 0:
            self.belief_white /= total_w
        else:
            for x in range(n):
                for y in range(n):
                    if (x + y) % 2 == 0:
                        self.belief_white[x, y] = 1.0
            self.belief_white /= self.belief_white.sum()

        if total_b > 0:
            self.belief_black /= total_b
        else:
            for x in range(n):
                for y in range(n):
                    if (x + y) % 2 == 1:
                        self.belief_black[x, y] = 1.0
            self.belief_black /= self.belief_black.sum()

    @staticmethod
    def _sensor_region(trap_loc: Tuple[int, int], cur_loc: Tuple[int, int]) -> str:
        tx, ty = trap_loc
        cx, cy = cur_loc
        dx = abs(tx - cx)
        dy = abs(ty - cy)

        if dx == 0 and dy == 0:
            return "far"  # standing on trap teleports; treat as far for sensors

        if max(dx, dy) == 1:
            return "diag" if dx == 1 and dy == 1 else "edge"
        if max(dx, dy) == 2:
            return "outer"
        return "far"

    @staticmethod
    def _likelihood(region: str, heard: bool, felt: bool) -> float:
        if region == "edge":
            p_h, p_f = 0.50, 0.30
        elif region == "diag":
            p_h, p_f = 0.25, 0.15
        elif region == "outer":
            p_h, p_f = 0.10, 0.00
        else:
            p_h, p_f = 0.0, 0.0

        prob = 1.0
        prob *= p_h if heard else (1.0 - p_h)
        prob *= p_f if felt else (1.0 - p_f)
        return prob

    def _normalize_belief(self, belief: np.ndarray, parity: int) -> None:
        s = float(belief.sum())
        if s > 0.0:
            belief /= s
        else:
            n = self.board_size
            for x in range(n):
                for y in range(n):
                    belief[x, y] = 1.0 if (x + y) % 2 == parity else 0.0
            belief /= belief.sum()

    def _update_beliefs_from_sensors(
        self, current_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]
    ) -> None:
        (heard_w, felt_w), (heard_b, felt_b) = sensor_data

        # White trap
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x + y) % 2 != 0:
                    continue
                region = self._sensor_region((x, y), current_loc)
                like = self._likelihood(region, heard_w, felt_w)
                self.belief_white[x, y] *= like
        self._normalize_belief(self.belief_white, parity=0)

        # Black trap
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x + y) % 2 != 1:
                    continue
                region = self._sensor_region((x, y), current_loc)
                like = self._likelihood(region, heard_b, felt_b)
                self.belief_black[x, y] *= like
        self._normalize_belief(self.belief_black, parity=1)

    def _record_teleport_if_any(self, board: "board.Board") -> None:
        current_loc = board.chicken_player.get_location()

        if (
            self.expected_next_location is not None
            and current_loc == self.spawn
            and self.prev_location is not None
            and self.prev_location != self.spawn
        ):
            trap_loc = self.expected_next_location
            x, y = trap_loc
            parity = (x + y) % 2

            if parity == 0:
                self.belief_white[:, :] = 0.0
                self.belief_white[x, y] = 1.0
                self._normalize_belief(self.belief_white, parity=0)
            else:
                self.belief_black[:, :] = 0.0
                self.belief_black[x, y] = 1.0
                self._normalize_belief(self.belief_black, parity=1)

        self.expected_next_location = None

    # ------------------------------------------------------------------
    # Geometry / helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _next_location(
        loc: Tuple[int, int], direction: Direction
    ) -> Tuple[int, int]:
        x, y = loc
        if direction == Direction.UP:
            return (x, y - 1)
        if direction == Direction.DOWN:
            return (x, y + 1)
        if direction == Direction.LEFT:
            return (x - 1, y)
        if direction == Direction.RIGHT:
            return (x + 1, y)
        return loc

    def _trap_probability(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return 0.0
        return float(self.belief_white[x, y] + self.belief_black[x, y])

    def _neighbor_trap_probability(self, loc: Tuple[int, int]) -> float:
        x, y = loc
        best = 0.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                best = max(best, self._trap_probability((nx, ny)))
        return best

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _frontier_unvisited_neighbors(self, loc: Tuple[int, int]) -> int:
        x, y = loc
        count = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if (nx, ny) not in self.visited:
                    count += 1
        return count

    def _compute_cross_target(self) -> Tuple[int, int]:
        """
        Pick a far-side tile (roughly opposite spawn) of our parity,
        near the horizontal center. We bias toward getting to the far side
        early to avoid being walled off.
        """
        n = self.board_size
        sx, sy = self.spawn
        # Choose far row
        if sy < n // 2:
            target_y = n - 2  # near bottom
        else:
            target_y = 1      # near top

        # column near center with our parity
        cx = (n - 1) // 2
        # adjust parity
        if (cx + target_y) % 2 != self.my_parity:
            if cx + 1 < n:
                cx += 1
            else:
                cx -= 1
        return (cx, target_y)

    def _choose_egg_target(
        self, board: "board.Board", current_loc: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Pick a "good" future egg square: our parity, not yet egged, modest trap prob.
        """
        candidates: List[Tuple[int, int]] = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                pos = (x, y)
                if (x + y) % 2 != self.my_parity:
                    continue
                if pos in board.eggs_player:
                    continue
                if self._trap_probability(pos) > 0.55:
                    continue
                candidates.append(pos)

        if not candidates:
            return None

        return min(candidates, key=lambda p: self._manhattan(current_loc, p))

    # ------------------------------------------------------------------
    # Turd valuation
    # ------------------------------------------------------------------
    def _turd_value(
        self,
        board: "board.Board",
        current_loc: Tuple[int, int],
        turn: int,
    ) -> float:
        """
        Estimate how valuable it is to drop a turd on current_loc.
        High if:
        - near enemy
        - same row/column (wall)
        - on enemy egg color
        - somewhat central
        Penalized if very far from enemy (useless).
        """
        enemy_loc = board.chicken_enemy.get_location()
        dist_enemy = self._manhattan(current_loc, enemy_loc)

        # Far from enemy -> usually useless
        if dist_enemy > 6:
            return -self.useless_turd_penalty * 1.5

        value = 0.0

        # Close to enemy: more valuable
        if dist_enemy <= 2:
            value += 3.0
        elif dist_enemy <= 4:
            value += 1.5

        # Same row or column: good for making walls
        if current_loc[0] == enemy_loc[0] or current_loc[1] == enemy_loc[1]:
            value += 1.5

        # On enemy egg color parity
        if (current_loc[0] + current_loc[1]) % 2 == self.enemy_parity:
            value += self.enemy_parity_block_bonus

        # Centrality (lightly)
        cx = cy = (self.board_size - 1) / 2.0
        dist_center = abs(current_loc[0] - cx) + abs(current_loc[1] - cy)
        value += max(0.0, 3.0 - dist_center) * 0.5

        # Early wall-building slightly encouraged
        if turn < self.early_turd_cutoff_turn:
            value *= 1.2

        # Late game: extra if we still have many turds to spend
        turds_left = board.chicken_player.get_turds_left()
        if turn >= self.late_game_turn and turds_left > 0:
            value += 1.0

        return value

    # ------------------------------------------------------------------
    # Scoring (for non-EGG moves)
    # ------------------------------------------------------------------
    def _score_move_non_egg(
        self,
        board: "board.Board",
        move: Tuple[Direction, MoveType],
        target: Optional[Tuple[int, int]],
        risk_factor: float,
    ) -> float:
        direction, move_type = move
        current_loc = board.chicken_player.get_location()
        dest = self._next_location(current_loc, direction)
        turn = board.turn_count

        score = 0.0

        # Trap penalties (scaled by time + risk)
        turn_factor = max(0.4, 1.0 - turn / 60.0)
        trap_penalty = self.base_trap_penalty * turn_factor * risk_factor
        near_trap_penalty = self.base_near_trap_penalty * turn_factor * risk_factor

        p_trap = self._trap_probability(dest)
        score -= trap_penalty * p_trap

        near_prob = self._neighbor_trap_probability(dest)
        score -= near_trap_penalty * near_prob

        # Target attraction
        if target is not None:
            dist_now = self._manhattan(current_loc, target)
            dist_dest = self._manhattan(dest, target)
            score += self.target_weight * (dist_now - dist_dest)

        # Early cross-the-board drive
        dist_now_cross = self._manhattan(current_loc, self.cross_target)
        dist_dest_cross = self._manhattan(dest, self.cross_target)
        if turn < self.cross_early_turn:
            score += self.cross_weight_early * (dist_now_cross - dist_dest_cross)
        elif turn < self.cross_mid_turn:
            score += self.cross_weight_mid * (dist_now_cross - dist_dest_cross)

        # Turd-specific logic
        if move_type == MoveType.TURD:
            turd_value = self._turd_value(board, current_loc, turn)
            score += self.turd_block_weight * turd_value
            if current_loc == self.spawn:
                score -= 4.0

        # Exploration
        if dest not in self.visited:
            score += self.explore_bonus
        frontier_neighbors = self._frontier_unvisited_neighbors(dest)
        score += self.frontier_bonus_weight * frontier_neighbors

        # Central control (weak)
        cx = cy = (self.board_size - 1) / 2.0
        dist_center_dest = abs(dest[0] - cx) + abs(dest[1] - cy)
        score -= self.center_weight * dist_center_dest

        # Loop avoidance
        if self.prev_location is not None and dest == self.prev_location:
            score -= self.backtrack_penalty
        elif dest in self.recent_positions:
            score -= self.loop_penalty

        return score

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def play(
        self,
        board: "board.Board",
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        # Check for trapdoor teleport last turn
        self._record_teleport_if_any(board)

        # Update Bayesian beliefs
        current_loc = board.chicken_player.get_location()
        self._update_beliefs_from_sensors(current_loc, sensor_data)

        # Egg margin -> adaptive risk
        own_eggs = len(board.eggs_player)
        enemy_eggs = len(board.eggs_enemy)
        margin = own_eggs - enemy_eggs

        if margin < 0:
            risk_factor = 0.7   # behind -> more aggressive
            egg_trap_threshold = 0.9
        elif margin > 1:
            risk_factor = 1.1   # ahead -> safer
            egg_trap_threshold = 0.6
        else:
            risk_factor = 1.0
            egg_trap_threshold = 0.75

        moves = board.get_valid_moves()
        if not moves:
            return (Direction.UP, MoveType.PLAIN)

        egg_moves = [m for m in moves if m[1] == MoveType.EGG]
        turd_moves = [m for m in moves if m[1] == MoveType.TURD]
        plain_moves = [m for m in moves if m[1] == MoveType.PLAIN]

        # --------------------------------------------------------------
        # 1. EGGS FIRST: choose best reasonably safe egg move.
        # --------------------------------------------------------------
        safe_egg_moves: List[Tuple[Direction, MoveType]] = []
        for m in egg_moves:
            direction, _ = m
            dest = self._next_location(current_loc, direction)
            if self._trap_probability(dest) < egg_trap_threshold:
                safe_egg_moves.append(m)

        if safe_egg_moves:
            best_egg = None
            best_score = float("-inf")

            for m in safe_egg_moves:
                direction, _ = m
                dest = self._next_location(current_loc, direction)
                score = 0.0

                # Egg reward
                score += self.egg_reward

                # Exploration & frontier
                if dest not in self.visited:
                    score += self.explore_bonus
                frontier_neighbors = self._frontier_unvisited_neighbors(dest)
                score += self.frontier_bonus_weight * frontier_neighbors

                # Centrality
                cx = cy = (self.board_size - 1) / 2.0
                dist_center_dest = abs(dest[0] - cx) + abs(dest[1] - cy)
                score -= self.center_weight * dist_center_dest

                # Early cross drive
                turn = board.turn_count
                dist_now_cross = self._manhattan(current_loc, self.cross_target)
                dist_dest_cross = self._manhattan(dest, self.cross_target)
                if turn < self.cross_early_turn:
                    score += self.cross_weight_early * (dist_now_cross - dist_dest_cross)
                elif turn < self.cross_mid_turn:
                    score += self.cross_weight_mid * (dist_now_cross - dist_dest_cross)

                # Light trap penalty
                score -= 10.0 * self._trap_probability(dest) * risk_factor

                # Avoid trivial backtracking
                if self.prev_location is not None and dest == self.prev_location:
                    score -= 1.5

                if score > best_score:
                    best_score = score
                    best_egg = m

            if best_egg is not None:
                direction, move_type = best_egg
                self.prev_location = current_loc
                self.expected_next_location = self._next_location(current_loc, direction)
                self.visited.add(current_loc)
                self.recent_positions.append(current_loc)
                if len(self.recent_positions) > 6:
                    self.recent_positions.pop(0)
                return best_egg

        # --------------------------------------------------------------
        # 2. No eggs available: choose best plain/turd move.
        #    Early game, avoid turds unless they're very good.
        # --------------------------------------------------------------
        turn = board.turn_count
        if turn < self.early_turd_cutoff_turn:
            # we will still consider turds, but only if *no* plain moves exist
            non_turd_moves = [m for m in moves if m[1] != MoveType.TURD]
            candidate_moves = non_turd_moves if non_turd_moves else moves
        else:
            candidate_moves = moves

        target = self._choose_egg_target(board, current_loc)

        best_move = None
        best_score = float("-inf")

        for m in candidate_moves:
            score = self._score_move_non_egg(board, m, target, risk_factor)
            if score > best_score:
                best_score = score
                best_move = m

        if best_move is None:
            best_move = moves[0]

        direction, move_type = best_move

        # Update memory
        self.prev_location = current_loc
        self.expected_next_location = self._next_location(current_loc, direction)
        self.visited.add(current_loc)
        self.recent_positions.append(current_loc)
        if len(self.recent_positions) > 6:
            self.recent_positions.pop(0)

        return best_move
