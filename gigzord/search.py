from chess import polyglot

from gigzord.evaluation import evaluate_board
from gigzord.move_ordering import get_sorted_moves
from timeout import Timeout
import time
import logging

logger = logging.getLogger(__name__)
transposition_table = {}
known_positions = {}
nodes_explored = 0


def get_best_move(board, max_move_time, max_depth):
    """iterative deepening with timeout"""
    global nodes_explored
    global known_positions
    board_copy = board.copy()
    nodes_explored = 0
    start_time = time.time()

    # do best book move
    with polyglot.open_reader("engines/Human.bin") as reader:
        for entry in reader.find_all(board):
            logger.info("Using move from opening book")
            board_copy.push(entry.move)
            zobrist_hash = polyglot.zobrist_hash(board_copy)
            known_positions[zobrist_hash] = True
            return None, entry.move

    # iterative deepening
    depth = 0
    score = 0

    moves = get_sorted_moves(board)
    with Timeout(max_move_time):
        try:
            while depth < max_depth:
                depth += 1
                score, moves = negamax(board, depth, board.turn)
                logger.info(f"Score: {score}, PV: {[str(m) for m in moves]}")
                if score > 10000:
                    break
        except TimeoutError:
            pass
    board_copy.push(moves[0])
    zobrist_hash = polyglot.zobrist_hash(board_copy)
    known_positions[zobrist_hash] = True

    logger.info(f"Nodes per second: {round(nodes_explored / (time.time() - start_time))}")
    if score == float('-inf'):
        return score, get_sorted_moves(board)[0]
    return score, moves[0]


def negamax(board, depth, alpha=float('-inf'), beta=float('inf')):
    global nodes_explored
    global known_positions

    zobrist_hash = polyglot.zobrist_hash(board)
    # if we've already been in this position on the actual game board, this is essentially a draw
    if zobrist_hash in known_positions:
        return 0, []

    # check transposition table if best move has already been calculated for this board position
    if zobrist_hash in transposition_table:
        tt_score, tt_pv, tt_depth = transposition_table[zobrist_hash]
        if depth <= tt_depth:
            return tt_score, tt_pv

    # recursive base case
    if depth == 0:
        return quiesce(board, alpha, beta), []
    best_score = float('-inf')
    best_path = []

    # move ordering for more effective pruning
    moves = get_sorted_moves(board)
    for move in moves:

        # "pretend" to make move and calculate score
        board.push(move)
        nodes_explored += 1
        score, path = negamax(board, depth - 1,  -beta, -alpha)
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

    transposition_table[zobrist_hash] = (best_score, best_path, depth)
    return best_score, best_path


def quiesce(board, alpha, beta):
    global nodes_explored

    stand_pat = evaluate_board(board)
    if abs(stand_pat) > 10000:
        return stand_pat
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            nodes_explored += 1
            score = -quiesce(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha
