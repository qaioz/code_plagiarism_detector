"""
Plagiarized from neetcode/0002-add-two-numbers.py.

Same dummy-node + carry-propagation linked-list addition. Renamed dummy ->
head, cur -> tail; flattened the Solution class wrapper into a free function;
inlined the val % 10 / val // 10 split into two assignments.
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def add_two_numbers(l1, l2):
    head = ListNode()
    tail = head
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        total = v1 + v2 + carry
        carry = total // 10
        tail.next = ListNode(total % 10)
        tail = tail.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return head.next
