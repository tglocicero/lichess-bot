from gigzord.piece_evals import piece_value


def get_sorted_moves(board):
    moves = list(board.legal_moves)
    capture_moves = [move for move in moves if board.is_capture(move)]
    sorted_moves = sorted(capture_moves, key=lambda move: static_exchange_evaluation(board, move), reverse=True)
    non_capture_moves = [move for move in moves if not board.is_capture(move)]
    sorted_moves.extend(non_capture_moves)
    return sorted_moves


def static_exchange_evaluation(board, move) -> int:
    """
    Perform static exchange evaluation (SEE) on a move to determine if it is a net gain or loss in material.
    """
    board.push(move)
    target_square = move.to_square
    attacking_side = not board.turn
    gain = piece_value[board.piece_type_at(target_square)]  # Gain from capturing the piece on the target square

    # Simulate exchanges on the target square
    material_balance = gain
    attackers = board.attackers(attacking_side, target_square)
    attackers = sorted(attackers, key=lambda square: piece_value[board.piece_type_at(square)])

    while attackers:
        attacker_square = attackers.pop(0)
        attacking_piece_value = piece_value[board.piece_type_at(attacker_square)]

        material_balance -= attacking_piece_value  # Cost of losing the attacker

        if material_balance < 0:
            break  # If we're losing material, stop

        attacking_side = not attacking_side
        attackers = board.attackers(attacking_side, target_square)
        attackers = sorted(attackers, key=lambda square: piece_value[board.piece_type_at(square)])

    board.pop()  # Restore board to original state
    return material_balance
