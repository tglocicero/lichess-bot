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
MAX_MOVE_TIME = 10


class GigZordEngine(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_transposition_table = {}
        self.search_transposition_table = {}
        self.killer_moves = {}  # For killer move heuristic
        self.history_heuristic = {}  # For history heuristic

        # Initialize piece values for MVV-LVA
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

    def eval_board(self, board: chess.Board) -> float:
        zobrist_key = polyglot.zobrist_hash(board)
        memoized_eval = self.eval_transposition_table.get(zobrist_key)
        if memoized_eval is not None:
            return memoized_eval * (1 if board.turn else -1)

        # Material evaluation
        white_material = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in self.piece_values.items()
        )
        black_material = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in self.piece_values.items()
        )
        material_score = white_material - black_material

        # Mobility evaluation
        is_white_turn = board.turn
        board.turn = chess.WHITE
        white_mobility = board.legal_moves.count()
        board.turn = chess.BLACK
        black_mobility = board.legal_moves.count()
        board.turn = is_white_turn
        mobility_score = 0.1 * (white_mobility - black_mobility)

        evaluation = material_score + mobility_score
        self.eval_transposition_table[zobrist_key] = evaluation
        return evaluation * (1 if board.turn else -1)

    def ordered_moves(self, board, captures_only=False, depth=0):
        moves = list(board.legal_moves)
        if captures_only:
            moves = [move for move in moves if board.is_capture(move)]

        # MVV-LVA ordering for captures
        def mvv_lva(move):
            if board.is_capture(move):
                attacker_piece = board.piece_at(move.from_square)
                if attacker_piece is None:
                    attacker_value = 0
                else:
                    attacker_value = self.piece_values.get(attacker_piece.piece_type, 0)

                # Handle en passant captures
                if board.is_en_passant(move):
                    # The captured pawn is one rank behind the to_square
                    if board.turn == chess.WHITE:
                        victim_square = move.to_square - 8
                    else:
                        victim_square = move.to_square + 8
                    victim_piece = board.piece_at(victim_square)
                else:
                    victim_piece = board.piece_at(move.to_square)

                if victim_piece is None:
                    victim_value = 0
                else:
                    victim_value = self.piece_values.get(victim_piece.piece_type, 0)

                return (victim_value * 10 - attacker_value)
            else:
                return 0  # Non-captures

        # Prioritize killer moves
        killer_moves = self.killer_moves.get(depth, [])

        # Prioritize history heuristic scores
        def history_score(move):
            return self.history_heuristic.get(move.uci(), 0)

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

        best_score = float('-inf')
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
                self.history_heuristic[move.uci()] = self.history_heuristic.get(
                    move.uci(), 0) + 2 ** depth

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

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        start_time = time.time()
        depth = 1
        max_depth = 5

        self.best_move_found = None
        move, score = self.negamax(board, depth, float('-inf'), float('inf'), max_depth)
        logger.info(f" - depth {depth}: {move}, {score}")
        with Timeout(MAX_MOVE_TIME):
            try:
                while depth < max_depth:
                    depth += 1
                    move, score = self.negamax(board, depth, float('-inf'), float('inf'), max_depth)
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
