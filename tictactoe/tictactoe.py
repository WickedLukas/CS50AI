"""
Tic Tac Toe Player
"""

import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0
    for row in range(3):
        for col in range(3):
            if board[row][col] == X:
                x_count += 1
            elif board[row][col] == O:
                o_count += 1
    if (x_count - o_count) == 0:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                actions.add((row, col))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY:
        raise ValueError

    result = copy.deepcopy(board)
    result[action[0]][action[1]] = player(board)
    return result


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = [row for row in board]
    for i in range(3):
        column = []
        for j in range(3):
            column.append(board[j][i])
        lines.append(column)
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2-i] for i in range(3)])

    for line in lines:
        if line == [X] * 3:
            return X
        elif line == [O] * 3:
            return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:
        return True

    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    temp = winner(board)
    if temp == X:
        return 1
    elif temp == O:
        return -1
    else:
        return 0


def min_value(board, last_min=-1):
    if terminal(board):
        return utility(board)

    board_actions = actions(board)
    action = board_actions.pop()
    result_value = max_value(result(board, action))

    if (result_value <= last_min):
        return result_value

    for action in board_actions:
        value = min(result_value, max_value(result(board, action)), result_value)
        if value < result_value:
            result_value = value
            if (result_value <= last_min):
                return result_value

    return result_value


def max_value(board, last_max=1):
    if terminal(board):
        return utility(board)

    board_actions = actions(board)
    action = board_actions.pop()
    result_value = min_value(result(board, action))

    if (result_value >= last_max):
        return result_value

    for action in board_actions:
        value = max(result_value, min_value(result(board, action)), result_value)
        if value > result_value:
            result_value = value
            if (result_value >= last_max):
                return result_value

    return result_value


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    board_actions = actions(board)
    result_action = board_actions.pop()
    if player(board) == X:
        result_value = min_value(result(board, result_action))
        if (result_value == 1):
            return result_action
    else:
        result_value = max_value(result(board, result_action))
        if (result_value == -1):
            return result_action

    for action in board_actions:
        if player(board) == X:
            value = min_value(result(board, action), result_value)
            if value > result_value:
                result_value = value
                result_action = action
                if (result_value == 1):
                    return result_action
        else:
            value = max_value(result(board, action), result_value)
            if value < result_value:
                result_value = value
                result_action = action
                if (result_value == -1):
                    return result_action

    return result_action
