"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import time

import chess
from chess import polyglot
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.types import MOVE, HOMEMADE_ARGS_TYPE
import logging

from timeout import Timeout

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)
VALUE_RANKED_PIECES = [chess.QUEEN, chess.ROOK,
                       chess.BISHOP, chess.KNIGHT, chess.PAWN]
MAX_MOVE_TIME = 90


class GigZordEngine(MinimalEngine):
    transposition_table = {}

    def eval_board(self, board: chess.Board) -> float:
        zobrist_key = polyglot.zobrist_hash(board)
        memoized_eval = self.transposition_table.get(zobrist_key)
        if memoized_eval is not None:
            return memoized_eval * (1 if board.turn else -1)

        num_white_kings = len(board.pieces(chess.KING, chess.WHITE))
        num_white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
        num_white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
        num_white_bishops_and_knights = len(board.pieces(
            chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.WHITE))
        num_white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))

        num_black_kings = len(board.pieces(chess.KING, chess.BLACK))
        num_black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
        num_black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
        num_black_bishops_and_knights = len(board.pieces(
            chess.BISHOP, chess.BLACK)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        num_black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))

        material_score = (
            200 * (num_white_kings - num_black_kings)
            + 9 * (num_white_queens - num_black_queens)
            + 5 * (num_white_rooks - num_black_rooks)
            + 3 * (num_white_bishops_and_knights -
                   num_black_bishops_and_knights)
            + 1 * (num_white_pawns - num_black_pawns)
        )

        is_white_turn = board.turn
        board.turn = True
        white_mobility = board.legal_moves.count()
        board.turn = False
        black_mobility = board.legal_moves.count()
        board.turn = is_white_turn
        mobility_score = 0.1 * (white_mobility - black_mobility)

        evaluation = material_score + mobility_score
        self.transposition_table[zobrist_key] = evaluation
        return (material_score + mobility_score) * (1 if board.turn else -1)

    def ordered_moves(self, board, captures_only=False):
        moves = list(board.legal_moves)
        if captures_only:
            moves = [move for move in moves if board.is_capture(move)]
        # Optionally sort moves to improve efficiency
        return moves

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        stand_pat = self.eval_board(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in self.ordered_moves(board, captures_only=True):
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> (chess.Move, float):
        if board.is_checkmate():
            return None, float('-inf') + (self.max_depth - depth)
        if depth == 0:
            return None, self.quiescence_search(board, alpha, beta)

        best_score = float('-inf')
        best_move = None
        for move in self.ordered_moves(board):
            board.push(move)
            _, score = self.negamax(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_move, best_score

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        start_time = time.time()
        depth = 1
        max_depth = 5

        self.best_move_found = None
        move, score = self.negamax(board, depth, float('-inf'), float('inf'))
        logger.info(f" - depth {depth}: {move}, {score}")
        with Timeout(MAX_MOVE_TIME):
            try:
                while depth < max_depth:
                    depth += 1
                    move, score = self.negamax(
                        board, depth, float('-inf'), float('inf'))
                    self.best_move_found = move
                    logger.info(f" - depth {depth}: {move}, {score}")
                    if time.time() - start_time < MAX_MOVE_TIME / 100 and depth > 3:
                        max_depth += 1
            except TimeoutError:
                depth -= 1

        logger.info(f"Best Move: {move}")
        logger.info(f"Time: {round(time.time() - start_time, 2)} seconds")
        logger.info(f"Depth: {depth}")
        logger.info(f"Eval: {score}")
        return PlayResult(move, ponder=None)


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(
                time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(
                time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(
                time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(
                time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(
            root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)
