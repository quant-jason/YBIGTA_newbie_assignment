from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")

class SegmentTree(Generic[T, U]):
    def __init__(self, data: list, u: str):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.u = u

        #sum 이외에 쓸 일은 없었다고 한다.
        if self.u == "sum":
            self.cal = lambda a, b: a + b
            self.default = 0
        elif self.u == "max":
            self.cal = lambda a, b: max(a, b)
            self.default = -10**9
        elif self.u == "min":
            self.cal = lambda a, b: min(a, b)
            self.default = 10**9
        else:
            ValueError("구현하세요!")

        self.build(data, 1, 0, self.n - 1)

    def build(self, arr, node, l, r):
        if l == r:
            self.tree[node] = arr[l]
            return
        mid = (l + r) // 2
        self.build(arr, 2 * node, l, mid)
        self.build(arr, 2 * node + 1, mid + 1, r)
        self.tree[node] = self.cal(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, l, r):
        return self._query(l, r, 1, 0, self.n - 1)

    def _query(self, l, r, node, nl, nr):
        if nl >= l and nr <= r:
            return self.tree[node]
        mid = (nl + nr) // 2
        if nr < l or nl > r:
            return self.default
        left_val = self._query(l, r, 2 * node, nl, mid)
        right_val = self._query(l, r, 2 * node + 1, mid + 1, nr)
        return self.cal(left_val, right_val)

    def update(self, idx, val):
        self._update(idx, val, 1, 0, self.n - 1)

    def _update(self, idx, val, node, l, r):
        if l == r:
            self.tree[node] = val
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(idx, val, 2 * node, l, mid)
        else:
            self._update(idx, val, 2 * node + 1, mid + 1, r)
        self.tree[node] = self.cal(self.tree[2 * node], self.tree[2 * node + 1])

    @staticmethod
    def find_kth(seg_tree: SegmentTree, k: int, l: int, r: int, node: int) -> int:

        if l == r:
            return l
        mid = (l + r) // 2
        left_sum = seg_tree.tree[node * 2]
        if left_sum >= k:
            return SegmentTree.find_kth(seg_tree, k, l, mid, node * 2)
        else:
            return SegmentTree.find_kth(seg_tree, k - left_sum, mid + 1, r, node * 2 + 1)

        


import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main():
    input_lines = sys.stdin.read().splitlines()
    t = int(input_lines[0])
    line_index = 1

    results = []

    for _ in range(t):
        n, m = map(int, input_lines[line_index].split())
        line_index += 1
        requests = list(map(int, input_lines[line_index].split()))
        line_index += 1

        size = n + m
        data = [0] * size

        position = {}
        for i in range(n):
            pos = m + i
            data[pos] = 1
            position[i + 1] = pos 

        seg_tree = SegmentTree(data, u="sum")
        next_top = m - 1  

        output = []
        for movie in requests:
            pos = position[movie]
            count_above = seg_tree.query(0, pos- 1)
            output.append(str(count_above))

            seg_tree.update(pos, 0)
            seg_tree.update(next_top, 1)
            position[movie] = next_top

            next_top -= 1

        results.append(" ".join(output))

    print("\n".join(results))


if __name__ == "__main__":
    main()