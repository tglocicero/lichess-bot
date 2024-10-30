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
MAX_MOVE_TIME = 30


class GigZordEngine(MinimalEngine):
    best_move_found = None
    transposition_table = {}
    max_depth = 6
    killer_moves = [[None, None] for _ in range(max_depth)]
    history_heuristic = {}

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
        return evaluation * (1 if board.turn else -1)

    def is_endgame(self, board):
        # if no queens, it's endgame
        return len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)) == 0

    def quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        zobrist_key = polyglot.zobrist_hash(board)
        tt_entry = self.transposition_table.get(zobrist_key)
        if tt_entry:
            tt_depth, tt_score, tt_flag = tt_entry
            if tt_depth >= 0:
                if tt_flag == 'exact':
                    return tt_score
                elif tt_flag == 'lower' and tt_score > alpha:
                    alpha = tt_score
                elif tt_flag == 'upper' and tt_score < beta:
                    beta = tt_score
                if alpha >= beta:
                    return tt_score

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

        # Store the final result in the TT as an exact score or lower bound
        flag = 'exact' if best_score > alpha else 'lower'
        self.transposition_table[zobrist_key] = (0, best_score, flag)
        return best_score

    def ordered_moves(self, board, depth):
        moves = list(board.legal_moves)

        # Separate captures from non-captures
        captures = [move for move in moves if board.is_capture(move)]
        non_captures = [move for move in moves if not board.is_capture(move)]

        # Sort captures by MVV-LVA
        def mvv_lva_orderer(move):
            if board.is_capture(move):
                victim_piece = board.piece_at(move.to_square)  # The piece being captured
                aggressor_piece = board.piece_at(move.from_square)  # The piece making the move
                if victim_piece and aggressor_piece:
                    victim_value = piece_value[victim_piece.piece_type]
                    aggressor_value = piece_value[aggressor_piece.piece_type]
                    return (victim_value, -aggressor_value)  # Sort by victim first, then aggressor
                return (0, 0)
        captures.sort(key=lambda move: mvv_lva_orderer(move), reverse=True)

        # Apply killer move and history heuristics
        killers = [move for move in non_captures if move in self.killer_moves[depth-1]]
        history_sorted = sorted(non_captures, key=lambda move: self.history_heuristic.get(move, 0), reverse=True)
        return captures + killers + history_sorted

    def update_killer_moves(self, depth, move):
        if move not in self.killer_moves[depth-1]:
            self.killer_moves[depth-1][1] = self.killer_moves[depth-1][0]
            self.killer_moves[depth-1][0] = move

    def update_history_heuristic(self, move, depth):
        if move not in self.history_heuristic:
            self.history_heuristic[move] = 0
            self.history_heuristic[move] += depth ** 2

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, prev_move=None) -> (chess.Move, int):
        zobrist_key = polyglot.zobrist_hash(board)
        tt_entry = self.transposition_table.get(zobrist_key)
        if tt_entry:
            tt_depth, tt_score, tt_flag = tt_entry
            if tt_depth >= depth:
                if tt_flag == 'exact':
                    return prev_move, tt_score
                elif tt_flag == 'lower' and tt_score > alpha:
                    alpha = tt_score
                elif tt_flag == 'upper' and tt_score < beta:
                    beta = tt_score
                if alpha >= beta:
                    return prev_move, tt_score

        if board.is_checkmate():
            return prev_move, float('inf') if board.turn else float('-inf')

        if board.is_stalemate():
            return prev_move, 0

        if depth == 0:
            return prev_move, self.quiescence(board, alpha, beta)

        best_score = float('-inf')
        best_move = None
        for move in self.ordered_moves(board, depth):
            board.push(move)
            score = -1 * self.negamax(board, depth - 1, -1 * beta, -1 * alpha, move)[1]
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move

            # Beta cutoff, so we store the move as a killer move if it's not a capture
            if best_score >= beta:
                self.update_history_heuristic(move, depth)
                if not board.is_capture(move):
                    self.update_killer_moves(depth, move)
                return best_move, best_score

            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        # Store the result in the transposition table
        if best_score <= alpha:
            flag = 'upper'  # The score is an upper bound
        elif best_score >= beta:
            flag = 'lower'  # The score is a lower bound
        else:
            flag = 'exact'  # The score is exact

        self.transposition_table[zobrist_key] = depth, best_score, flag
        self.best_move_found = best_move
        return best_move, best_score

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        with polyglot.open_reader("engines/book1.bin") as reader:
            for entry in reader.find_all(board):
                return PlayResult(entry.move, ponder=None)

        start_time = time.time()
        depth = 1

        self.best_move_found = None
        # killer_moves = [[None, None] for _ in range(self.max_depth)]
        move, score = self.negamax(board, depth, float('-inf'), float('inf'))
        logger.info(f" - depth {depth}: {move}, {score}")
        with Timeout(MAX_MOVE_TIME):
            try:
                while depth < self.max_depth:
                    depth += 1
                    move, score = self.negamax(board, depth, float('-inf'), float('inf'))
                    self.best_move_found = move
                    logger.info(f" - depth {depth}: {move}, {score}")
            except TimeoutError:
                depth -= 1

        move = self.ordered_moves(board, 1)[0] if move is None else move

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
