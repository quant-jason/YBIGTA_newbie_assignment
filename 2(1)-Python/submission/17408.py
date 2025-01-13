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

#"시간부족으로 한 문제 포기하겠습니다 ㅠㅠ"

class Pair():
    pass

def main() -> None:
    # 구현하세요!
    pass


if __name__ == "__main__":
    main()