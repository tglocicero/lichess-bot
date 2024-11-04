import chess
from chess import polyglot

from gigzord.evaluation import evaluate_board
from gigzord.move_ordering import get_sorted_moves
from timeout import Timeout
import logging

logger = logging.getLogger(__name__)
transposition_table = {}


def get_best_move(board, max_move_time, max_depth):
    """iterative deepening with timeout"""
    # do best book move
    with polyglot.open_reader("engines/Human.bin") as reader:
        for entry in reader.find_all(board):
            logger.info("Using move from opening book")
            return None, entry.move

    # iterative deepening
    depth = 0
    score = 0
    moves = get_sorted_moves(board)
    with Timeout(max_move_time):
        try:
            while depth < max_depth:
                depth += 1
                score, moves = negamax(board, depth, float('-inf'), float('inf'), board.turn)
                logger.info(f"Score: {score}, PV: {[str(m) for m in moves]}")
                if score == 20000:
                    break
        except TimeoutError:
            pass
    return score, moves[0]


def negamax(board, depth, alpha, beta, is_white):
    # check transposition table if best move has already been calculated for this board position
    zobrist_hash = polyglot.zobrist_hash(board)
    if (zobrist_hash, depth) in transposition_table and not board.is_repetition(2):
        tt_score, tt_pv = transposition_table[(zobrist_hash, depth)]
        return tt_score, tt_pv

    # recursive base case
    if depth == 0 or board.is_game_over():
        return quiesce(board, alpha, beta, is_white), []

    best_score = float('-inf')
    best_path = []

    # move ordering for more effective pruning
    moves = get_sorted_moves(board)
    for move in moves:

        # "pretend" to make move and calculate score
        board.push(move)
        score, path = negamax(board, depth - 1, -beta, -alpha, is_white)
        score = -score
        board.pop()

        # book keep best move and maintain principal variation
        if score > best_score:
            best_score = score
            best_path = [move] + path  # include current move in best path

        # alpha-beta pruning
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    transposition_table[(zobrist_hash, depth)] = (best_score, best_path)
    return best_score, best_path


def quiesce(board, alpha, beta, is_white):
    stand_pat = evaluate_board(board, is_white)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            score = -quiesce(board, -beta, -alpha, is_white)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha
