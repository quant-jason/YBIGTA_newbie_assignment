from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable



T = TypeVar("T")

@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False

class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[Iterable[T]]) -> None:
        for tokens in seq:
            curr_node = self[0]
            for tok in tokens:
                child_node = next((self[child] for child in curr_node.children if self[child].body == tok), None)
                if not child_node:
                    child_node = TrieNode(body=tok)
                    self.append(child_node)
                    new_index = len(self) - 1
                    curr_node.children.append(new_index)
                curr_node = child_node
            curr_node.is_end = True

    @staticmethod
    def perm(n, mod=1000000007):
        if n == 0 or n == 1:
            return 1
        return (n * Trie.perm(n - 1, mod)) % mod
    
    @staticmethod
    def lcp_length(a: str, b: str) -> int:
        length = min(len(a), len(b))
        i = 0
        while i < length and a[i] == b[i]:
            i += 1
        return i

    @staticmethod
    def lcp(str_list):
        str_list.sort()
        partial_insert = []
        length = len(str_list)

        for i in range(length):
            word = str_list[i]
            if not word:
                partial_insert.append([])
                continue
            lcp_left = 0
            lcp_right = 0
            if i > 0:
                lcp_left = Trie.lcp_length(word, str_list[i-1])
            if i < length - 1:
                lcp_right = Trie.lcp_length(word, str_list[i+1])
            L = max(lcp_left, lcp_right)
            cutoff = min(len(word), L + 1)
            prefix_ords = [ord(ch) for ch in word[:cutoff]]
            partial_insert.append(prefix_ords)
        
        return partial_insert

    def drill(self, node):
        stack = [node]
        results = []
        while stack:
            current = stack.pop()
            if current.children:
                a = len(current.children)
                b = 1 if current.is_end else 0
                results.append(Trie.perm(a + b))
                for item in current.children:
                    stack.append(self[item])
        return results

    @property
    def order_count(self):
        curr_node = self[0]
        vals = self.drill(curr_node)
        ans = 1
        for v in vals:
            ans *= v
        return ans




import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""




def main() -> None:
    lines = sys.stdin.readlines()
    n = int(lines[0].strip())
    str_list = [line.strip().replace(" ", "") for line in lines[1:1+n]]
    # print(str_list)
    p = Trie.lcp(str_list)

    trie1: Trie[int] = Trie()
    trie1.push(p)
    result = trie1.order_count
    print(result % 1000000007)

if __name__ == "__main__":
    main()

