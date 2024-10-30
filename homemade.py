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

from piece_evals import piece_value, pawnEvalWhite, knightEval, pawnEvalBlack, bishopEvalBlack, rookEvalBlack, \
    bishopEvalWhite, rookEvalWhite, queenEval, kingEvalEndGameWhite, kingEvalEndGameBlack, kingEvalWhite, kingEvalBlack
from timeout import Timeout

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)
VALUE_RANKED_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
MAX_MOVE_TIME = 10


class GigZordEngine(MinimalEngine):
    transposition_table = {}
    best_move_found = None


    def evaluate_piece(self, piece: chess.Piece, square: chess.Square, end_game: bool) -> int:
        piece_type = piece.piece_type
        mapping = []
        if piece_type == chess.PAWN:
            mapping = pawnEvalWhite if piece.color == chess.WHITE else pawnEvalBlack
        if piece_type == chess.KNIGHT:
            mapping = knightEval
        if piece_type == chess.BISHOP:
            mapping = bishopEvalWhite if piece.color == chess.WHITE else bishopEvalBlack
        if piece_type == chess.ROOK:
            mapping = rookEvalWhite if piece.color == chess.WHITE else rookEvalBlack
        if piece_type == chess.QUEEN:
            mapping = queenEval
        if piece_type == chess.KING:
            # use end game piece-square tables if neither side has a queen
            if end_game:
                mapping = (
                    kingEvalEndGameWhite
                    if piece.color == chess.WHITE
                    else kingEvalEndGameBlack
                )
            else:
                mapping = kingEvalWhite if piece.color == chess.WHITE else kingEvalBlack

        return mapping[square]


    def eval_board(self, board: chess.Board) -> float:
        zobrist_key = polyglot.zobrist_hash(board)
        memoized_eval = self.transposition_table.get(zobrist_key)
        if memoized_eval is not None:
            return memoized_eval * (1 if board.turn else -1)

        material_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
            value = piece_value[piece.piece_type] + self.evaluate_piece(piece, square, self.is_endgame(board))
            material_score += value if piece.color == chess.WHITE else -value

        is_white_turn = board.turn
        board.turn = True
        white_mobility = board.pseudo_legal_moves.count()
        board.turn = False
        black_mobility = board.pseudo_legal_moves.count()
        board.turn = is_white_turn
        mobility_score = white_mobility - black_mobility

        evaluation = material_score + mobility_score
        self.transposition_table[zobrist_key] = evaluation
        return evaluation * (1 if board.turn else -1)

    def move_value(self, board, move, is_endgame):
        if move.promotion is not None:
            return 3 if board.turn else -3

        piece_type = board.piece_at(move.from_square).piece_type
        if piece_type in [chess.PAWN, chess.KNIGHT]:
            return 2

        if piece_type in [chess.BISHOP]:
            return 1

        if piece_type in [chess.ROOK]:
            return 0

        if piece_type in [chess.QUEEN]:
            return -1

        return -2

    def is_endgame(self, board):
        # if no queens, it's endgame
        return len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)) == 0

    def ordered_moves(self, board):
        # https://www.chessprogramming.org/MVV-LVA
        moves = board.legal_moves
        if self.best_move_found and moves and self.best_move_found in moves:
            moves = list(board.legal_moves)
            moves.remove(self.best_move_found)
            moves = [self.best_move_found] + moves

        def orderer(move):
            return self.move_value(board, move, self.is_endgame(board))

        in_order = sorted(moves, key=orderer, reverse=board.turn)
        return in_order

    def quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        stand_pat = self.eval_board(board)
        best_score = stand_pat

        if best_score >= beta:
            return best_score
        if best_score > alpha:
            alpha = best_score

        # Generate all captures and checks
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha)
                board.pop()
                best_score = max(best_score, score)
                if best_score >= beta:
                    return best_score
                alpha = max(alpha, best_score)

        return best_score

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, prev_move=None) -> (chess.Move, int):
        if board.is_checkmate():
            return prev_move, float('inf') if board.turn else float('-inf')

        if board.is_stalemate():
            return prev_move, 0

        if depth == 0:
            return prev_move, self.quiescence(board, alpha, beta)

        # zobrist_key = polyglot.zobrist_hash(board)
        # memoized_eval = self.transposition_table.get(zobrist_key)
        # if memoized_eval is not None:
        #     return prev_move, memoized_eval

        best_score = float('-inf')
        best_move = None
        for move in self.ordered_moves(board):
            board.push(move)
            score = -1 * self.negamax(board, depth - 1, -1 * beta, -1 * alpha, move)[1]
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_move, best_score

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        self.transposition_table = {}
        start_time = time.time()
        depth = 1
        max_depth = 4

        self.best_move_found = None
        move, score = self.negamax(board, depth, float('-inf'), float('inf'))
        logger.info(f" - depth {depth}: {move}, {score}")
        with Timeout(MAX_MOVE_TIME):
            try:
                while depth < max_depth:
                    depth += 1
                    move, score = self.negamax(board, depth, float('-inf'), float('inf'))
                    self.best_move_found = move
                    logger.info(f" - depth {depth}: {move}, {score}")
            except TimeoutError:
                depth -= 1

        if move is None:
            move, score = self.negamax(board, 1, float('-inf'), float('inf'))

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
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)
