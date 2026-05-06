"""
Negative example. A fixed-capacity circular (ring) buffer with O(1) push
and pop, overwriting the oldest element when full. Different from the
indexed pyrival data structures (BitArray, DSU, FenwickTree) — different
access pattern (FIFO with wraparound), different invariants, different
operations.
"""


class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def push(self, value):
        if self.size == self.capacity:
            self.head = (self.head + 1) % self.capacity
        else:
            self.size += 1
        self.buffer[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity

    def pop(self):
        if self.size == 0:
            raise IndexError("pop from empty buffer")
        value = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return value

    def peek(self):
        if self.size == 0:
            raise IndexError("peek into empty buffer")
        return self.buffer[self.head]

    def __len__(self):
        return self.size
