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
logger = logging.getLogger(__name__)

# Maximum time (in seconds) the bot will think before making a move
MAX_MOVE_TIME = 10

# Piece indices for PeSTO
PIECE_TYPES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

# Material values for middle-game and end-game
MG_VALUE = [82, 337, 365, 477, 1025,  0]  # Middle-game values
EG_VALUE = [94, 281, 297, 512,  936,  0]  # End-game values

# Game phase increments
GAME_PHASE_INC = [0, 1, 1, 2, 4, 0]

# Piece-square tables for middle-game and end-game
mg_pawn_table = [
     0,   0,   0,   0,   0,   0,  0,   0,
    98, 134,  61,  95,  68, 126, 34, -11,
    -6,   7,  26,  31,  65,  56, 25, -20,
   -14,  13,   6,  21,  23,  12, 17, -23,
   -27,  -2,  -5,  12,  17,   6, 10, -25,
   -26,  -4,  -4, -10,   3,   3, 33, -12,
   -35,  -1, -20, -23, -15,  24, 38, -22,
     0,   0,   0,   0,   0,   0,  0,   0,
]

eg_pawn_table = [
     0,   0,   0,   0,   0,   0,   0,   0,
   178, 173, 158, 134, 147, 132, 165, 187,
    94, 100,  85,  67,  56,  53,  82,  84,
    32,  24,  13,   5,  -2,   4,  17,  17,
    13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     4,   7,  -6,   1,   0,  -5,  -1,  -8,
    13,   8,   8,  10,  13,   0,   2,  -7,
     0,   0,   0,   0,   0,   0,   0,   0,
]

mg_knight_table = [
  -167, -89, -34, -49,  61, -97, -15, -107,
   -73, -41,  72,  36,  23,  62,   7,  -17,
   -47,  60,  37,  65,  84, 129,  73,   44,
    -9,  17,  19,  53,  37,  69,  18,   22,
   -13,   4,  16,  13,  28,  19,  21,   -8,
   -23,  -9,  12,  10,  19,  17,  25,  -16,
   -29, -53, -12,  -3,  -1,  18, -14,  -19,
  -105, -21, -58, -33, -17, -28, -19,  -23,
]

eg_knight_table = [
   -58, -38, -13, -28, -31, -27, -63, -99,
   -25,  -8, -25,  -2,  -9, -25, -24, -52,
   -24, -20,  10,   9,  -1,  -9, -19, -41,
   -17,   3,  22,  22,  22,  11,   8, -18,
   -18,  -6,  16,  25,  16,  17,   4, -18,
   -23,  -3,  -1,  15,  10,  -3, -20, -22,
   -42, -20, -10,  -5,  -2, -20, -23, -44,
   -29, -51, -23, -15, -22, -18, -50, -64,
]

mg_bishop_table = [
   -29,   4, -82, -37, -25, -42,   7,  -8,
   -26,  16, -18, -13,  30,  59,  18, -47,
   -16,  37,  43,  40,  35,  50,  37,  -2,
    -4,   5,  19,  50,  37,  37,   7,  -2,
    -6,  13,  13,  26,  34,  12,  10,   4,
     0,  15,  15,  15,  14,  27,  18,  10,
     4,  15,  16,   0,   7,  21,  33,   1,
   -33,  -3, -14, -21, -13, -12, -39, -21,
]

eg_bishop_table = [
   -14, -21, -11,  -8, -7,  -9, -17, -24,
    -8,  -4,   7, -12, -3, -13,  -4, -14,
     2,  -8,   0,  -1, -2,   6,   0,   4,
    -3,   9,  12,   9, 14,  10,   3,   2,
    -6,   3,  13,  19,  7,  10,  -3,  -9,
   -12,  -3,   8,  10, 13,   3,  -7, -15,
   -14, -18,  -7,  -1,  4,  -9, -15, -27,
   -23,  -9, -23,  -5, -9, -16,  -5, -17,
]

mg_rook_table = [
    32,  42,  32,  51, 63,  9,  31,  43,
    27,  32,  58,  62, 80, 67,  26,  44,
    -5,  19,  26,  36, 17, 45,  61,  16,
   -24, -11,   7,  26, 24, 35,  -8, -20,
   -36, -26, -12,  -1,  9, -7,   6, -23,
   -45, -25, -16, -17,  3,  0,  -5, -33,
   -44, -16, -20,  -9, -1, 11,  -6, -71,
   -19, -13,   1,  17, 16,  7, -37, -26,
]

eg_rook_table = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
]

mg_queen_table = [
   -28,   0,  29,  12,  59,  44,  43,  45,
   -24, -39,  -5,   1, -16,  57,  28,  54,
   -13, -17,   7,   8,  29,  56,  47,  57,
   -27, -27, -16, -16,  -1,  17,  -2,   1,
    -9, -26,  -9, -10,  -2,  -4,   3,  -3,
   -14,   2, -11,  -2,  -5,   2,  14,   5,
   -35,  -8,  11,   2,   8,  15,  -3,   1,
    -1, -18,  -9,  10, -15, -25, -31, -50,
]

eg_queen_table = [
    -9,  22,  22,  27,  27,  19,  10,  20,
   -17,  20,  32,  41,  58,  25,  30,   0,
   -20,   6,   9,  49,  47,  35,  19,   9,
     3,  22,  24,  45,  57,  40,  57,  36,
   -18,  28,  19,  47,  31,  34,  39,  23,
   -16, -27,  15,   6,   9,  17,  10,   5,
   -22, -23, -30, -16, -16, -23, -36, -32,
   -33, -28, -22, -43,  -5, -32, -20, -41,
]

mg_king_table = [
   -65,  23,  16, -15, -56, -34,   2,  13,
    29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -9,  24,   2, -16, -20,   6,  22, -22,
   -17, -20, -12, -27, -30, -25, -14, -36,
   -49,  -1, -27, -39, -46, -44, -33, -51,
   -14, -14, -22, -46, -44, -30, -15, -27,
     1,   7,  -8, -64, -43, -16,   9,   8,
   -15,  36,  12, -54,   8, -28,  24,  14,
]

eg_king_table = [
   -74, -35, -18, -18, -11,  15,   4, -17,
   -12,  17,  14,  17,  17,  38,  23,  11,
    10,  17,  23,  15,  20,  45,  44,  13,
    -8,  22,  24,  27,  26,  33,  26,   3,
   -18,  -4,  21,  24,  27,  23,   9, -11,
   -19,  -3,  11,  21,  23,  16,   7,  -9,
   -27, -11,   4,  13,  14,   4,  -5, -17,
   -53, -34, -21, -11, -28, -14, -24, -43,
]

# Organize the PSTs
MG_PESTO_TABLE = [
    mg_pawn_table,
    mg_knight_table,
    mg_bishop_table,
    mg_rook_table,
    mg_queen_table,
    mg_king_table
]

EG_PESTO_TABLE = [
    eg_pawn_table,
    eg_knight_table,
    eg_bishop_table,
    eg_rook_table,
    eg_queen_table,
    eg_king_table
]


class GigZordEngine(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_transposition_table = {}
        self.search_transposition_table = {}
        self.killer_moves = {}  # For killer move heuristic
        self.history_heuristic = {}  # For history heuristic
        self.infinity = 1000000
        self.best_move_found = None

    def evaluate_board(self, board: chess.Board) -> float:
        zobrist_key = polyglot.zobrist_hash(board)
        memoized_eval = self.eval_transposition_table.get(zobrist_key)
        if memoized_eval is not None:
            return memoized_eval

        mg = [0, 0]  # Middle-game scores for WHITE and BLACK
        eg = [0, 0]  # End-game scores for WHITE and BLACK
        game_phase = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color = 0 if piece.color == chess.WHITE else 1
                piece_type = piece.piece_type
                idx = PIECE_TYPES[piece_type]
                position = square if piece.color == chess.WHITE else chess.square_mirror(square)

                # Add material and PST values
                mg[color] += MG_VALUE[idx] + MG_PESTO_TABLE[idx][position]
                eg[color] += EG_VALUE[idx] + EG_PESTO_TABLE[idx][position]

                # Accumulate game phase
                game_phase += GAME_PHASE_INC[idx]

        # Total phase (max 24 as per PeSTO's implementation)
        total_phase = 24
        if game_phase > total_phase:
            game_phase = total_phase

        # Calculate phase weights
        mg_phase = game_phase
        eg_phase = total_phase - game_phase

        # Compute final evaluation
        mg_score = mg[0] - mg[1]
        eg_score = eg[0] - eg[1]
        score = (mg_score * mg_phase + eg_score * eg_phase) / total_phase

        # Adjust score based on side to move
        evaluation = score if board.turn == chess.WHITE else -score

        # Store in transposition table
        self.eval_transposition_table[zobrist_key] = evaluation

        return evaluation

    def ordered_moves(self, board, captures_only=False, depth=0):
        moves = list(board.legal_moves)
        if captures_only:
            moves = [move for move in moves if board.is_capture(move)]

        # MVV-LVA ordering for captures
        def mvv_lva(move):
            if board.is_capture(move):
                attacker_piece = board.piece_at(move.from_square)
                attacker_value = MG_VALUE[PIECE_TYPES[attacker_piece.piece_type]]

                # Handle en passant captures
                if board.is_en_passant(move):
                    victim_piece = chess.Piece(chess.PAWN, not board.turn)
                else:
                    victim_piece = board.piece_at(move.to_square)

                victim_value = MG_VALUE[PIECE_TYPES[victim_piece.piece_type]]

                return (victim_value * 10 - attacker_value)
            else:
                return 0  # Non-captures

        # Prioritize killer moves
        killer_moves = self.killer_moves.get(depth, [])

        # Prioritize history heuristic scores
        def history_score(move):
            return self.history_heuristic.get(move, 0)

        # Prioritize the best move from the transposition table
        zobrist_key = polyglot.zobrist_hash(board)
        tt_entry = self.search_transposition_table.get(zobrist_key)
        tt_move = tt_entry['best_move'] if tt_entry else None

        moves.sort(key=lambda move: (
            move == tt_move,                   # Transposition table move
            move in killer_moves,              # Killer move heuristic
            mvv_lva(move),                     # MVV-LVA heuristic
            history_score(move)                # History heuristic
        ), reverse=True)

        return moves

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        stand_pat = self.evaluate_board(board)
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

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, max_depth: int) -> (chess.Move, float):
        zobrist_key = polyglot.zobrist_hash(board)
        tt_entry = self.search_transposition_table.get(zobrist_key)

        if tt_entry is not None and tt_entry['depth'] >= depth:
            node_type = tt_entry['type']
            tt_score = tt_entry['score']
            if node_type == 'exact':
                return tt_entry['best_move'], tt_score
            elif node_type == 'lowerbound':
                alpha = max(alpha, tt_score)
            elif node_type == 'upperbound':
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_entry['best_move'], tt_score

        if depth == 0 or board.is_game_over():
            score = self.quiescence_search(board, alpha, beta)
            return None, score

        alpha_original = alpha  # Store the original alpha value

        # Null-move pruning conditions
        if depth >= 3 and not board.is_check() and not board.has_legal_en_passant() and len(board.piece_map()) > 12:
            R = 2  # Reduction value; commonly set to 2
            board.push(chess.Move.null())
            # Perform a null-move search with reduced depth
            _, score = self.negamax(board, depth - 1 - R, -beta, -beta + 1, max_depth)
            score = -score
            board.pop()
            if score >= beta:
                return None, beta  # Beta cutoff

        best_score = -self.infinity
        best_move = None

        for move in self.ordered_moves(board, depth=depth):
            board.push(move)
            _, score = self.negamax(board, depth - 1, -beta, -alpha, max_depth)
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            # Update history heuristic
            if depth == max_depth:
                self.history_heuristic[move] = self.history_heuristic.get(move, 0) + 2 ** depth

            alpha = max(alpha, score)
            if alpha >= beta:
                # Record killer move
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].append(move)
                    if len(self.killer_moves[depth]) > 2:  # Keep only top 2 killer moves
                        self.killer_moves[depth].pop(0)
                break

        # Store in transposition table
        entry = {
            'depth': depth,
            'score': best_score,
            'best_move': best_move
        }
        if best_score <= alpha_original:
            entry['type'] = 'upperbound'
        elif best_score >= beta:
            entry['type'] = 'lowerbound'
        else:
            entry['type'] = 'exact'
        self.search_transposition_table[zobrist_key] = entry

        return best_move, best_score

    def search(self, board: chess.Board, *args) -> PlayResult:
        start_time = time.time()
        depth = 1
        max_depth = 10

        self.best_move_found = None
        move, score = self.negamax(board, depth, -self.infinity, self.infinity, max_depth)
        logger.info(f" - depth {depth}: {move}, {score}")
        with Timeout(MAX_MOVE_TIME):
            try:
                while depth < max_depth:
                    depth += 1
                    move, score = self.negamax(board, depth, -self.infinity, self.infinity, max_depth)
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
