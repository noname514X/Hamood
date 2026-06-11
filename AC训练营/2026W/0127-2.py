import sys
import re
import ast


class ListNode:
    def __init__(self, x=0, next=None):
        self.val = x
        self.next = next


def reverseKGroup(head, k):
    if k <= 1 or not head:
        return head
    dummy = ListNode(0, head)
    prev = dummy
    while True:
        node = prev
        for _ in range(k):
            node = node.next
            if not node:
                return dummy.next
        tail = prev.next
        curr = tail.next
        for _ in range(k - 1):
            nxt = curr.next
            curr.next = prev.next
            prev.next = curr
            tail.next = nxt
            curr = nxt
        prev = tail


def build_list(arr):
    dummy = ListNode()
    cur = dummy
    for v in arr:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def to_pylist(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out


def main():
    data = sys.stdin.read()

    m_list = re.search(r"\[[\s\S]*?\]", data)
    head = []
    if m_list:
        try:
            head = ast.literal_eval(m_list.group(0))
        except Exception:
            head = []

    m = re.search(r"k\s*=\s*(-?\d+)", data)
    if m:
        k = int(m.group(1))
    else:
        lines = [l.strip() for l in data.strip().splitlines() if l.strip()]
        if len(lines) >= 2:
            try:
                maybe = lines[1]
                k = int(maybe)
            except Exception:
                k = 1
        else:
            k = 1
    n = build_list(head)
    res = reverseKGroup(n, k)
    print(to_pylist(res))


if __name__ == '__main__':
    main()