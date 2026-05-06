"""
Plagiarized from pyrival/BitArray.py.

Same bytearray-backed bit array with the same bit-shift accessors. Renamed
bytes -> data, kept __getitem__ / __setitem__ identical (same shift/mask).
"""


class BitVector:
    """implements bitarray using bytearray"""

    def __init__(self, size):
        self.data = bytearray((size >> 3) + 1)

    def __getitem__(self, index):
        return (self.data[index >> 3] >> (index & 7)) & 1

    def __setitem__(self, index, value):
        if value:
            self.data[index >> 3] |= 1 << (index & 7)
        else:
            self.data[index >> 3] &= ~(1 << (index & 7))
