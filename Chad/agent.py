from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional

import numpy as np

from game import *
from game.enums import Direction, MoveType


# ----------------------------------------------------------------------
# Lightweight state for alpha-beta search (no dependency on Board.apply_move)
# ----------------------------------------------------------------------
@dataclass
class LightState:
    my_loc: Tuple[int, int]
    enemy_loc: Tuple[int, int]
    my_eggs: Set[Tuple[int, int]]
    enemy_eggs: Set[Tuple[int, int]]
    my_turds: Set[Tuple[int, int]]
    enemy_turds: Set[Tuple[int, int]]
    my_turds_left: int
    enemy_turds_left: int
    turn: int


class PlayerAgent:
    """
    Chad v7 + alpha-beta – Territory racer + smart walls + Bayesian traps + tactical lookahead.

    Core Chad v7 behavior:
    - Bayesian beliefs for both trapdoors (white/black) updated from sensors.
    - Eggs-first policy: if a reasonably safe EGG move exists, always prefer it.
    - Strong early-game drive to cross to the far side of the board and fan out.
    - Frontier exploration bonus (dest with many unvisited neighbors).
    - Turds:
        * Rewarded only when near the enemy and wall-like (same row/col,
          enemy egg color, central).
        * Penalized when far from the enemy (useless turds).
        * Slight extra push to use remaining turds late game.

    NEW: Alpha-beta search on top of Chad v7:
    - Build a lightweight internal state copied from the real Board.
    - Use Chad's heuristic to order moves.
    - Run alpha-beta to a small depth over this light state:
        * Leaf eval uses egg diff + territory (reachable area) + trap risk.
    - At the root, alpha-beta is used as a tie-breaker among Chad's best moves.
    """

    # ==================================================================
    # Initialization
    # ==================================================================
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

        # --- Strategy parameters (Chad v7) -----------------------------
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

        # --- Alpha-beta search parameters ------------------------------
        self.search_depth: int = 2          # full plies (us, enemy)
        self.search_top_moves: int = 3      # only search top-K moves per node
        self._ab_time_left: Optional[Callable] = None

    # ==================================================================
    # Belief helpers
    # ==================================================================
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

    # ==================================================================
    # Geometry / helpers
    # ==================================================================
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

    # ==================================================================
    # Turd valuation
    # ==================================================================
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

    # ==================================================================
    # Scoring (for non-EGG moves) – Chad v7
    # ==================================================================
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

    # ==================================================================
    # Alpha-beta helpers: build & simulate light state
    # ==================================================================
    def _build_light_state(self, board: "board.Board") -> LightState:
        return LightState(
            my_loc=board.chicken_player.get_location(),
            enemy_loc=board.chicken_enemy.get_location(),
            my_eggs=set(board.eggs_player),
            enemy_eggs=set(board.eggs_enemy),
            my_turds=set(board.turds_player),
            enemy_turds=set(board.turds_enemy),
            my_turds_left=board.chicken_player.get_turds_left(),
            enemy_turds_left=board.chicken_enemy.get_turds_left(),
            turn=board.turn_count,
        )

    def _reachable_area(self, state: LightState, start: Tuple[int, int]) -> int:
        """
        BFS to count reachable tiles from 'start', treating eggs & turds as walls.
        """
        n = self.board_size
        walls = state.my_eggs | state.enemy_eggs | state.my_turds | state.enemy_turds

        if not (0 <= start[0] < n and 0 <= start[1] < n):
            return 0
        if start in walls:
            return 0

        visited: Set[Tuple[int, int]] = set()
        stack = [start]
        visited.add(start)

        while stack:
            x, y = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                loc = (nx, ny)
                if loc in visited or loc in walls:
                    continue
                visited.add(loc)
                stack.append(loc)

        return len(visited)

    def _light_get_legal_moves(
        self, state: LightState, maximizing: bool
    ) -> List[Tuple[Direction, MoveType]]:
        """
        Approximate legal moves in the light state:
        - Can't move off-board.
        - Can't move into eggs or turds (treat as walls).
        - Can always choose PLAIN or EGG, TURD if turds_left > 0.
        """
        n = self.board_size
        walls = state.my_eggs | state.enemy_eggs | state.my_turds | state.enemy_turds

        loc = state.my_loc if maximizing else state.enemy_loc
        turds_left = state.my_turds_left if maximizing else state.enemy_turds_left

        moves: List[Tuple[Direction, MoveType]] = []

        for d in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
            x, y = loc
            nx, ny = self._next_location(loc, d)
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            dest = (nx, ny)
            if dest in walls:
                continue

            # We include all three move types here. Alpha-beta + ordering will prune.
            moves.append((d, MoveType.PLAIN))
            moves.append((d, MoveType.EGG))
            if turds_left > 0:
                moves.append((d, MoveType.TURD))

        return moves

    def _simulate_move_state(
        self,
        state: LightState,
        move: Tuple[Direction, MoveType],
        maximizing: bool,
    ) -> LightState:
        """
        Apply a move to the light state. No trapdoors, no sensors; just
        movement + eggs/turds.
        """
        d, mtype = move

        # Copy everything
        my_eggs = set(state.my_eggs)
        enemy_eggs = set(state.enemy_eggs)
        my_turds = set(state.my_turds)
        enemy_turds = set(state.enemy_turds)

        my_loc = state.my_loc
        enemy_loc = state.enemy_loc
        my_turds_left = state.my_turds_left
        enemy_turds_left = state.enemy_turds_left

        if maximizing:
            old_loc = my_loc
            new_loc = self._next_location(my_loc, d)
            my_loc = new_loc
            if mtype == MoveType.EGG:
                my_eggs.add(old_loc)
            elif mtype == MoveType.TURD:
                my_turds.add(old_loc)
                my_turds_left = max(0, my_turds_left - 1)
        else:
            old_loc = enemy_loc
            new_loc = self._next_location(enemy_loc, d)
            enemy_loc = new_loc
            if mtype == MoveType.EGG:
                enemy_eggs.add(old_loc)
            elif mtype == MoveType.TURD:
                enemy_turds.add(old_loc)
                enemy_turds_left = max(0, enemy_turds_left - 1)

        return LightState(
            my_loc=my_loc,
            enemy_loc=enemy_loc,
            my_eggs=my_eggs,
            enemy_eggs=enemy_eggs,
            my_turds=my_turds,
            enemy_turds=enemy_turds,
            my_turds_left=my_turds_left,
            enemy_turds_left=enemy_turds_left,
            turn=state.turn + 1,
        )

    # ==================================================================
    # Evaluation & alpha-beta
    # ==================================================================
    def _evaluate_state(self, state: LightState) -> float:
        """
        Evaluate a light state from *our* perspective (maximizing player).
        """
        score = 0.0

        # Egg difference
        egg_diff = len(state.my_eggs) - len(state.enemy_eggs)
        score += egg_diff * self.egg_reward

        # Territory difference (reachable tiles from each chicken)
        my_reach = self._reachable_area(state, state.my_loc)
        enemy_reach = self._reachable_area(state, state.enemy_loc)
        score += 0.35 * (my_reach - enemy_reach)

        # Encourage progress toward cross_target (roughly the far side)
        dist_cross = self._manhattan(state.my_loc, self.cross_target)
        # Bigger board -> bigger baseline. Clamp at 0.
        score += 0.2 * (self.board_size * 2 - dist_cross)

        # Light centrality and trap risk
        cx = cy = (self.board_size - 1) / 2.0
        dist_center = abs(state.my_loc[0] - cx) + abs(state.my_loc[1] - cy)
        score -= 0.03 * dist_center

        score -= 15.0 * self._trap_probability(state.my_loc)

        return score

    def _alpha_beta(
        self,
        state: LightState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        # Soft time cutoff
        if self._ab_time_left is not None and self._ab_time_left() < 0.03:
            return self._evaluate_state(state)

        if depth == 0:
            return self._evaluate_state(state)

        moves = self._light_get_legal_moves(state, maximizing)
        if not moves:
            return self._evaluate_state(state)

        # Order moves using Chad's heuristic idea: approximate by egg vs non-egg preference
        # and distance toward cross target.
        def move_order_key(m: Tuple[Direction, MoveType]) -> float:
            d, mt = m
            loc = state.my_loc if maximizing else state.enemy_loc
            dest = self._next_location(loc, d)
            base = 0.0
            if mt == MoveType.EGG:
                base += self.egg_reward
            dist_cross = self._manhattan(dest, self.cross_target)
            base += 0.5 * (self.board_size * 2 - dist_cross)
            return base

        moves_sorted = sorted(moves, key=move_order_key, reverse=True)
        moves_sorted = moves_sorted[: self.search_top_moves]

        if maximizing:
            value = float("-inf")
            for m in moves_sorted:
                child = self._simulate_move_state(state, m, maximizing=True)
                value = max(
                    value,
                    self._alpha_beta(child, depth - 1, alpha, beta, False),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for m in moves_sorted:
                child = self._simulate_move_state(state, m, maximizing=False)
                value = min(
                    value,
                    self._alpha_beta(child, depth - 1, alpha, beta, True),
                )
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def _choose_with_alpha_beta(
        self,
        board: "board.Board",
        candidate_moves: List[Tuple[Direction, MoveType]],
    ) -> Tuple[Direction, MoveType]:
        """
        Use alpha-beta as a tie-breaker among Chad's candidate moves.
        """
        if not candidate_moves:
            return (Direction.UP, MoveType.PLAIN)
        if len(candidate_moves) == 1 or self.search_depth <= 0:
            return candidate_moves[0]

        root_state = self._build_light_state(board)
        best_move = candidate_moves[0]
        best_value = float("-inf")

        # Limit to top-K by simple heuristic ordering from current board
        cur_loc = board.chicken_player.get_location()
        target = self._choose_egg_target(board, cur_loc)
        scored: List[Tuple[float, Tuple[Direction, MoveType]]] = []

        for m in candidate_moves:
            direction, mtype = m
            dest = self._next_location(cur_loc, direction)
            score = 0.0

            if mtype == MoveType.EGG:
                score += self.egg_reward
            else:
                score += self._score_move_non_egg(board, m, target, risk_factor=1.0)

            # Light preference for moves that get closer to cross_target
            dist_cross = self._manhattan(dest, self.cross_target)
            score += 0.4 * (self.board_size * 2 - dist_cross)

            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        ordered_moves = [m for _, m in scored[: self.search_top_moves]]

        for m in ordered_moves:
            child = self._simulate_move_state(root_state, m, maximizing=True)
            val = self._alpha_beta(
                child,
                depth=self.search_depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=False,
            )
            if val > best_value:
                best_value = val
                best_move = m

        return best_move

    # ==================================================================
    # Main entry point
    # ==================================================================
    def play(
        self,
        board: "board.Board",
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        # Make time_left accessible to alpha-beta
        self._ab_time_left = time_left

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
        #    Then alpha-beta tie-breaks between good eggs.
        # --------------------------------------------------------------
        safe_egg_moves: List[Tuple[Direction, MoveType]] = []
        for m in egg_moves:
            direction, _ = m
            dest = self._next_location(current_loc, direction)
            if self._trap_probability(dest) < egg_trap_threshold:
                safe_egg_moves.append(m)

        if safe_egg_moves:
            # Let alpha-beta break ties between safe egg moves.
            chosen_egg = self._choose_with_alpha_beta(board, safe_egg_moves)

            direction, move_type = chosen_egg
            self.prev_location = current_loc
            self.expected_next_location = self._next_location(current_loc, direction)
            self.visited.add(current_loc)
            self.recent_positions.append(current_loc)
            if len(self.recent_positions) > 6:
                self.recent_positions.pop(0)
            return chosen_egg

        # --------------------------------------------------------------
        # 2. No eggs available: choose best plain/turd move.
        #    Early game, avoid turds unless they're very good.
        #    Use alpha-beta among the filtered candidates.
        # --------------------------------------------------------------
        turn = board.turn_count
        if turn < self.early_turd_cutoff_turn:
            # we will still consider turds, but only if *no* plain moves exist
            non_turd_moves = [m for m in moves if m[1] != MoveType.TURD]
            candidate_moves = non_turd_moves if non_turd_moves else moves
        else:
            candidate_moves = moves

        target = self._choose_egg_target(board, current_loc)

        # Use Chad v7 scoring to pre-filter: keep top K for alpha-beta
        scored_candidates: List[Tuple[float, Tuple[Direction, MoveType]]] = []
        for m in candidate_moves:
            if m[1] == MoveType.TURD:
                # rough score: treat as non-egg but allow turd logic
                score = self._score_move_non_egg(board, m, target, risk_factor)
            elif m[1] == MoveType.EGG:
                # shouldn't happen here (no egg moves), but safe-guard
                score = self.egg_reward
            else:
                score = self._score_move_non_egg(board, m, target, risk_factor)
            scored_candidates.append((score, m))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        ordered_candidates = [m for _, m in scored_candidates[: max(self.search_top_moves, 1)]]

        # If alpha-beta is disabled or only one move, just take best greedy move.
        if len(ordered_candidates) == 1 or self.search_depth <= 0:
            best_move = ordered_candidates[0]
        else:
            best_move = self._choose_with_alpha_beta(board, ordered_candidates)

        direction, move_type = best_move

        # Update memory
        self.prev_location = current_loc
        self.expected_next_location = self._next_location(current_loc, direction)
        self.visited.add(current_loc)
        self.recent_positions.append(current_loc)
        if len(self.recent_positions) > 6:
            self.recent_positions.pop(0)

        return best_move
