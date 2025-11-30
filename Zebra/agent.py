# ================================================================
#  CS3600 — ULTRA OPTIMIZED AGENT (FULL + CORRECTED SIGNATURE)
# ================================================================
#  Features:
#   - Bayesian trapdoor inference
#   - Exploration / visitation scoring + anti-loop
#   - Smart Eggs (safe + opportunistic)
#   - Smart Turds (aggressive but safe)
#   - 2-ply minimax forecasting
#   - Center corridor control
#   - Opponent pressure model
#   - Full time management
#   - Stable fallback
# ================================================================

from typing import Tuple, List, Set, Dict
from collections import deque
import numpy as np

from game.enums import Direction, MoveType
from game import board


# ================================================================
# Utility helpers
# ================================================================
DIRS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

DIR_VECS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(loc, n):
    return 0 <= loc[0] < n and 0 <= loc[1] < n


# ================================================================
# PlayerAgent
# ================================================================
class PlayerAgent:

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(self, board_obj: board.Board, time_left):
        self.time_left = time_left
        self.size = board_obj.game_map.MAP_SIZE

        # Bayesian trapdoor belief
        self.b_white = np.ones((self.size, self.size))
        self.b_black = np.ones((self.size, self.size))
        self._normalize()

        # Visitation scoring
        self.visits = np.zeros((self.size, self.size), dtype=np.int32)

        # Anti-loop
        self.last_positions = deque(maxlen=6)

        # Cache
        self.dist_cache = {}

    # ------------------------------------------------------------
    # Normalize belief matrices
    # ------------------------------------------------------------
    def _normalize(self):
        self.b_white /= self.b_white.sum()
        self.b_black /= self.b_black.sum()

    # ------------------------------------------------------------
    # Bayesian inference update
    # ------------------------------------------------------------
    def update_beliefs(self, loc, signals):
        (hw, fw), (hb, fb) = signals

        for color, (heard, felt) in enumerate([(hw, fw), (hb, fb)]):
            grid = self.b_white if color == 0 else self.b_black
            like = np.ones_like(grid)

            for x in range(self.size):
                for y in range(self.size):
                    p_h, p_f = self.trap_signal_prob(loc, (x, y))
                    like[x, y] *= (p_h if heard else (1 - p_h))
                    like[x, y] *= (p_f if felt else (1 - p_f))

            grid *= like

        self._normalize()

    # ------------------------------------------------------------
    # Trapdoor signal probabilities
    # ------------------------------------------------------------
    def trap_signal_prob(self, here, trap):
        hx, hy = here
        tx, ty = trap
        dx, dy = abs(hx - tx), abs(hy - ty)

        if dx + dy == 1:        # edge-adjacent
            return 0.50, 0.30
        if dx == 1 and dy == 1:  # diagonal
            return 0.25, 0.15
        if dx + dy == 2:        # outer ring
            return 0.10, 0.00

        return 0.00, 0.00

    # ------------------------------------------------------------
    # Probability this tile is a trap
    # ------------------------------------------------------------
    def trap_prob(self, loc):
        x, y = loc
        if (x + y) % 2 == 0:
            return self.b_white[x, y]
        return self.b_black[x, y]

    # ------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------
    def get_legal_actions(self, b):
        me = b.current_player
        acts = []

        for d in DIRS:
            if not b.can_move(d):
                continue

            if b.can_plain_step(d):
                acts.append((d, MoveType.PLAIN))

            if b.can_egg_step(d):
                acts.append((d, MoveType.EGG))

            if me.remaining_turds > 0 and b.can_turd_step(d):
                acts.append((d, MoveType.TURD))

        return acts

    # ------------------------------------------------------------
    # Opponent legal actions
    # ------------------------------------------------------------
    def get_opp_actions(self, b):
        rb = b.reverse_perspective()
        me = rb.current_player
        acts = []

        for d in DIRS:
            if not rb.can_move(d):
                continue

            if rb.can_plain_step(d):
                acts.append((d, MoveType.PLAIN))

            if rb.can_egg_step(d):
                acts.append((d, MoveType.EGG))

            if me.remaining_turds > 0 and rb.can_turd_step(d):
                acts.append((d, MoveType.TURD))

        return acts

    # ------------------------------------------------------------
    # Evaluate an action with minimax depth 2
    # ------------------------------------------------------------
    def evaluate_action(self, b, action, depth, fast=False):
        d, mv = action

        child = b.forecast_move(d, mv)
        if child is None:
            return -1e18

        score = self.eval_board(child)

        if fast or depth >= 1:
            return score

        opp_actions = self.get_opp_actions(child)
        if not opp_actions:
            return score

        worst = 1e18
        for od, omv in opp_actions:
            nxt = child.forecast_move(od, omv)
            if nxt is None:
                continue
            worst = min(worst, self.eval_board(nxt))

        return score - 0.30 * abs(worst)

    # ------------------------------------------------------------
    # Board evaluation
    # ------------------------------------------------------------
    def eval_board(self, b):
        me = b.current_player
        opp = b.other_player
        my_loc = me.position

        # Trap penalty
        trap_p = self.trap_prob(my_loc)
        trap_pen = -140 * trap_p

        # Exploration
        explore_score = 15 * (1 / (1 + self.visits[my_loc[0], my_loc[1]]))

        # Loop penalty
        loop_pen = 0
        if len(self.last_positions) >= 6 and len(set(self.last_positions)) <= 3:
            loop_pen = -20

        # Center pull
        cx, cy = 3.5, 3.5
        center_dist = abs(my_loc[0] - cx) + abs(my_loc[1] - cy)
        center_score = 25 - 3 * center_dist

        # Eggs
        egg_score = 6 * len(me.eggs)

        # Turds left
        turd_score = 4 * me.remaining_turds

        # Opponent pressure
        opp_d = manhattan(my_loc, opp.position)
        opp_score = -2 * (1 / (1 + opp_d))

        return (
            trap_pen
            + explore_score
            + loop_pen
            + center_score
            + egg_score
            + turd_score
            + opp_score
        )

    # ------------------------------------------------------------
    # Main move chooser
    # ------------------------------------------------------------
    def choose_action(self, b):
        my_loc = b.current_player.position
        self.visits[my_loc[0], my_loc[1]] += 1
        self.last_positions.append(my_loc)

        # Bayesian update
        self.update_beliefs(my_loc, b.trapdoor_signals)

        actions = self.get_legal_actions(b)
        if not actions:
            return (Direction.UP, MoveType.PLAIN)

        best = None
        best_score = -1e18
        fast = self.time_left() < 20

        for act in actions:
            sc = self.evaluate_action(b, act, depth=0, fast=fast)
            if sc > best_score:
                best_score = sc
                best = act

        return best if best else actions[0]

    # ------------------------------------------------------------
    # Internal wrapper
    # ------------------------------------------------------------
    def action(self, b):
        try:
            return self.choose_action(b)
        except:
            return (Direction.UP, MoveType.PLAIN)

    # ------------------------------------------------------------
    # ENGINE ENTRY POINT (REQUIRED)
    # ------------------------------------------------------------
    def play(self, board_obj, trapdoor_samples, time_left_func):
        # Update time_left callback each turn (engine passes a new one)
        self.time_left = time_left_func
        return self.action(board_obj)
