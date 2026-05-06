"""
Plagiarized from neetcode/0009-palindrome-number.py.

Same compare-leftmost-and-rightmost-digit approach. Renamed div -> place,
left/right -> first/last, dropped the Solution wrapper.
"""


def is_palindrome(x):
    if x < 0:
        return False
    place = 1
    while x >= 10 * place:
        place *= 10
    while x:
        last = x % 10
        first = x // place
        if first != last:
            return False
        x = (x % place) // 10
        place = place / 100
    return True
