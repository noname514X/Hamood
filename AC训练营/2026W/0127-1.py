import sys
import re
import ast


class ListNode:
    def __init__(self, x=0, next=None):
        self.val = x
        self.next = next


def mergeTwoLists(l1, l2):
    dummy = ListNode()
    tail = dummy
    a, b = l1, l2
    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next


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
    lists = re.findall(r"\[.*?\]", data)
    if not lists:
        print([])
        return
    try:
        l1 = ast.literal_eval(lists[0])
    except Exception:
        l1 = []
    if len(lists) > 1:
        try:
            l2 = ast.literal_eval(lists[1])
        except Exception:
            l2 = []
    else:
        l2 = []
    n1 = build_list(l1)
    n2 = build_list(l2)
    res = mergeTwoLists(n1, n2)
    print(to_pylist(res))


if __name__ == '__main__':
    main()
