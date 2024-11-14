import chess

from gigzord.piece_evals import piece_value, pawnEvalWhite, knightEval, bishopEvalWhite, rookEvalWhite, queenEval, \
    pawnEvalBlack, bishopEvalBlack, rookEvalBlack, kingEvalEndGameWhite, kingEvalEndGameBlack, kingEvalWhite, \
    kingEvalBlack


def get_is_endgame(board):
    return len(board.piece_map()) <= 16


def evaluate_board(board, is_white):
    """material score that considers board position of pieces"""
    # if board.is_checkmate():
    #     if not is_white:
    #         return float('inf') * (-1 if board.turn else 1)
    #     return float('inf') * (1 if board.turn else -1)

    is_endgame = get_is_endgame(board)
    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        value = piece_value[piece.piece_type] + evaluate_piece(piece, square, is_endgame)
        material_score += value if piece.color == chess.WHITE else -value
    return material_score * (1 if board.turn else -1)


def evaluate_piece(piece, square, is_endgame) -> int:
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
        if is_endgame:
            mapping = kingEvalEndGameWhite if piece.color == chess.WHITE else kingEvalEndGameBlack
        else:
            mapping = kingEvalWhite if piece.color == chess.WHITE else kingEvalBlack

    return mapping[square]
