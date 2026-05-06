"""
Negative example. Validates a 9x9 sudoku board: no duplicate digit in any
row, column, or 3x3 sub-box. Different problem family than any indexed
neetcode entry — sudoku grid validation is unique to this case.
"""


def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for r in range(9):
        for c in range(9):
            value = board[r][c]
            if value == '.':
                continue
            box = (r // 3) * 3 + (c // 3)
            if value in rows[r] or value in cols[c] or value in boxes[box]:
                return False
            rows[r].add(value)
            cols[c].add(value)
            boxes[box].add(value)
    return True
